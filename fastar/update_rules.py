from functools import reduce
from operator import and_
import jax.lax as lax
from jax.ops import index_update
import jax.numpy as np
from jax.interpreters import xla
import jax.scipy.special as special
import numpy as onp
from jax import vjp
from jax.abstract_arrays import make_shaped_array
from jax.ad_util import zeros_like_aval
from jax.util import curry, safe_zip, safe_map, unzip2, WrapHashably

import fastar.util as util
from . import interpreter as fa

map = safe_map
zip = safe_zip

init_value = 0


def _unbroadcast_slice(s, shape):
    return tuple(s if dim_sz > 1 else slice(None)
                 for s, dim_sz in zip(s[len(s) - len(shape):], shape))

def sliceableop_update(func, old_out, output_mask,
                       input_slices_from_output_slice, *args, **params):
    """Update rule for operations where any slice of their (only) output can be
    calculated by applying the operation itself to one slice of each input.

    :param input_slices_from_output_slice:
        Specifies which input slices are required for an output slice."""

    outval, old_outmask = old_out
    new_mask = output_mask & ~old_outmask
    output_slices = util.mask_to_slices(new_mask)
    for output_slice in output_slices:
        assert np.all(outval[output_slice] == init_value)

        input_slices = input_slices_from_output_slice(output_slice)
        sliced_inputs = tuple(arg[s] for arg, s in zip(args, input_slices))
        sliced_output = func.bind(*sliced_inputs, **params)
        outval = index_update(outval, output_slice, sliced_output)

    return fa.Parray((outval, output_mask))

# n-ary elementwise operators with broadcasting
@curry
def nop_update(op, ans, *args):
    args, arg_masks = zip(*args)
    ans, ans_mask = ans
    new_ans_mask = reduce(and_, arg_masks)
    slices = util.mask_to_slices(new_ans_mask &~ ans_mask)
    for s in slices:
        part_args = [a[_unbroadcast_slice(s, np.shape(a))] for a in args]
        ans = index_update(ans, s, op.bind(*part_args))
    return fa.Parray((ans, new_ans_mask))


nops = [
    lax.abs_p,
    lax.add_p,
    lax.ceil_p,
    lax.cos_p,
    lax.div_p,
    lax.eq_p,
    lax.exp_p,
    lax.floor_p,
    lax.ge_p,
    lax.gt_p,
    lax.le_p,
    lax.log_p,
    lax.log1p_p,
    lax.lt_p,
    lax.max_p,
    lax.min_p,
    lax.mul_p,
    lax.ne_p,
    lax.neg_p,
    lax.rem_p,
    lax.select_p,
    lax.sign_p,
    lax.sin_p,
    lax.sub_p,
    lax.tanh_p,
]

for op in nops:
    fa.update_rules[op] = nop_update(op)

# Logit and expit use custom_transforms so their primitives have a different
# form.
@curry
def logexpit_update(op, ans, x, **params):
    x, x_mask = x
    (ans, ans_mask), = ans
    slices = util.mask_to_slices(x_mask &~ ans_mask)
    for s in slices:
        part_x = x[_unbroadcast_slice(s, np.shape(x))]
        ans = index_update(ans, s, op.bind(part_x, **params)[0])
    return [fa.Parray((ans, x_mask))]

for op in [special.expit.prim, special.logit.prim]:
    fa.update_rules[op] = logexpit_update(op)


# Cheap ops, these are assumed to require little or no computation
@curry
def cheap_op_update(op, old_out, *args, **params):
    args, args_mask = zip(*args)
    # TODO: use onp equivalents to process the masks
    return fa.Parray((op.bind(*args, **params),
                      onp.bool_(op.bind(*args_mask, **params))))

cheap_ops = [
    lax.broadcast_p,
    lax.broadcast_in_dim_p,
    lax.concatenate_p,
    lax.convert_element_type_p,
    lax.reshape_p,
    lax.rev_p,
    lax.slice_p,
    lax.tie_in_p,
    lax.transpose_p,
]

for op in cheap_ops:
    fa.update_rules[op] = cheap_op_update(op)

def gather_update(
        old_out, operand, start_indices, dimension_numbers, slice_sizes,
        **params):
    # Treat gather as a cheap op, but need to handle start_indices correctly
    operand, operand_mask = operand
    start_indices, start_indices_mask = start_indices
    assert onp.all(start_indices_mask)
    # Treat gather as a cheap op
    return fa.parray(
        lax.gather_p.bind(
            operand, start_indices, dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes, **params),
        onp.asarray(lax.gather_p.bind(
            operand_mask, start_indices, dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes, **params)))
fa.update_rules[lax.gather_p] = gather_update

def gather_static_update(
        old_out, operand, start_indices, start_indices_shape, dimension_numbers,
        slice_sizes, **params):
    # Treat gather as a cheap op, but need to handle start_indices correctly
    operand, operand_mask = operand
    start_indices = onp.reshape(onp.array(start_indices, dtype=int),
                                start_indices_shape)
    # Treat gather as a cheap op
    return fa.parray(
        lax.gather_p.bind(
            operand, start_indices, dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes, **params),
        onp.asarray(lax.gather_p.bind(
            operand_mask, start_indices, dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes, **params)))
fa.update_rules[lax.gather_static_p] = gather_static_update

# Reductions
@curry
def reduce_update(op, ans, a, **params):
    a, a_mask = a
    ans, ans_mask = ans
    axes = params['axes']
    new_ans_mask = onp.all(a_mask, axes)
    slices = util.mask_to_slices(new_ans_mask &~ ans_mask)
    for s in slices:
        a_slice = list(s)
        for axis in axes:
            a_slice.insert(axis, slice(None))
        a_part = a[tuple(a_slice)]
        if 'input_shape' in params:
            params['input_shape'] = np.shape(a_part)
        ans = index_update(ans, s, op.bind(a_part, **params))
    return fa.Parray((ans, new_ans_mask))

reduce_ops = [
    lax.reduce_max_p,
    lax.reduce_min_p,
    lax.reduce_sum_p,
]

for op in reduce_ops:
    fa.update_rules[op] = reduce_update(op)


def dot_update(ans, a, b, precision=None):
    a, a_mask = a
    b, b_mask = b
    ansval, ansmask = ans
    if a.ndim == b.ndim == 1:
        # Vector-vector product
        if onp.all(a_mask) and onp.all(b_mask) and not ansmask:
            return fa.Parray((lax.dot(a, b, precision=precision), True))
        else:
            return ans
    elif a.ndim == 2 and b.ndim == 1:
        # Matrix-vector product
        if onp.all(b_mask):
            new_ansmask = onp.all(a_mask, 1)
            for s in util.mask_to_slices(new_ansmask &~ ansmask):
                ansval = index_update(
                    ansval, s, lax.dot_p.bind(a[s], b, precision=precision))
            return fa.Parray((ansval, new_ansmask))
        else:
            return ans
    elif a.ndim == 1 and b.ndim == 2:
        # Vector-matrix product
        if onp.all(a_mask):
            new_ansmask = onp.all(b_mask, 0)
            for s in util.mask_to_slices(new_ansmask &~ ansmask):
                ansval = index_update(
                    ansval, s, lax.dot_p.bind(a, b[:, s[0]],
                                              precision=precision))
            return fa.Parray((ansval, new_ansmask))
        else:
            return fa.Parray((ansval, ansmask))
    else:
        # Matrix-matrix product
        new_ansmask = onp.all(a_mask, 1, keepdims=True) & onp.all(b_mask, 0)
        for s in util.mask_to_slices(new_ansmask &~ ansmask):
            ansval = index_update(
                ansval, s, lax.dot_p.bind(
                    a[s[0]], b[:, s[1]], precision=precision))
        return fa.Parray((ansval, new_ansmask))

fa.update_rules[lax.dot_p] = dot_update


def dot_general_update(ans, a, b, dimension_numbers, precision=None):
    a, a_mask = a
    b, b_mask = b
    ansval, ansmask = ans
    (a_cont_dims, b_cont_dims), (a_btch_dims, b_btch_dims) = dimension_numbers

    a_outr_dims = tuple(d for d in range(a.ndim)
                        if d not in a_cont_dims + a_btch_dims)
    b_outr_dims = tuple(d for d in range(b.ndim)
                        if d not in b_cont_dims + b_btch_dims)

    cont_ndim = len(a_cont_dims)
    btch_ndim = len(a_btch_dims)

    # Batch dims to front
    a_mask = onp.moveaxis(a_mask, a_btch_dims, range(btch_ndim))
    a_cont_dims_ = tuple(c + sum(1 for b in a_btch_dims if b > c)
                         for c in a_cont_dims)
    b_mask = onp.moveaxis(b_mask, b_btch_dims, range(len(b_btch_dims)))
    b_cont_dims_ = tuple(c + sum(1 for b in b_btch_dims if b > c)
                         for c in b_cont_dims)

    a_mask = onp.all(a_mask, a_cont_dims_)
    b_mask = onp.all(b_mask, b_cont_dims_)

    new_ansmask = (
        a_mask[(Ellipsis,) + len(b_outr_dims) * (onp.newaxis,)]
        & b_mask[btch_ndim * (slice(None),)
                 + len(a_outr_dims) * (onp.newaxis,)])
    for s in util.mask_to_slices(new_ansmask &~ ansmask):
        s_btch, s_a, s_b = (s[:btch_ndim],
                            s[btch_ndim:btch_ndim + len(a_outr_dims)],
                            s[btch_ndim + len(a_outr_dims):])
        a_slice = tuple((s_btch + s_a + cont_ndim * (slice(None),))[d] for d in
                        onp.argsort(a_btch_dims + a_outr_dims + a_cont_dims))
        b_slice = tuple((s_btch + s_b + cont_ndim * (slice(None),))[d] for d in
                        onp.argsort(b_btch_dims + b_outr_dims + b_cont_dims))
        ansval = index_update(ansval, s,
                              lax.dot_general_p.bind(
                                  a[a_slice], b[b_slice],
                                  dimension_numbers=dimension_numbers,
                                  precision=precision))
    return fa.Parray((ansval, new_ansmask))

fa.update_rules[lax.dot_general_p] = dot_general_update


def pad_update(old_out, input, padding_value, padding_config):
    for (lo, hi, interior) in padding_config:
        if interior > 0 and (lo < 0 or hi < 0): raise NotImplementedError(
            "Interior and negative padding on same axis not yet implemented.")

    input, input_mask = input
    padding_value, padding_value_mask = padding_value
    outval, old_outmask = old_out

    unpad_slice = tuple(
        slice(lo if lo > 0 else None,
              -hi if hi > 0 else None,
              None if interior == 0 else interior + 1)
        for (lo, hi, interior) in padding_config)
    unpad = lambda x: x[unpad_slice]

    def pad_slices():
        for index, (lo, hi, interior) in enumerate(padding_config):
            def nonoverlapping(s):
                return unpad_slice[:index] + (s,) + \
                       tuple([slice(None)] * (old_outmask.ndim - index - 1))

            if lo > 0:
                yield nonoverlapping(slice(lo))

            for i in range(interior):
                yield nonoverlapping(slice(lo + i + 1, -hi, (interior + 1)))

            if hi > 0:
                yield nonoverlapping(slice(-hi, None))

    old_padding_value_mask = any(onp.any(old_outmask[s]) for s in pad_slices())
    new_padding_value_mask = padding_value_mask and not old_padding_value_mask

    output_mask = old_outmask.copy()

    input_crop_slice = tuple(
        slice(-lo if lo < 0 else None,
              hi if hi < 0 else None)
        for (lo, hi, _) in padding_config)

    cropped_input_mask = input_mask[input_crop_slice]
    output_mask[unpad_slice] = cropped_input_mask
    if new_padding_value_mask:
        for s in pad_slices():
            outval = index_update(outval, s, np.broadcast_to(padding_value,
                                                        output_mask[s].shape))
            output_mask[s] = True

    cropped_input_new_mask = cropped_input_mask & ~unpad(old_outmask)
    cropped_input_slices = util.mask_to_slices(cropped_input_new_mask)
    for cropped_input_slice in cropped_input_slices:
        output_slice = tuple(
            slice((lo if lo > 0 else 0) + s.start * (interior + 1),
                  (lo if lo > 0 else 0) + s.stop * (interior + 1),
                  interior + 1)
            for s, (lo, _, interior) in
            zip(cropped_input_slice, padding_config))

        input_slice = tuple(
            slice(s.start + (-lo if lo < 0 else 0),
                  s.stop + (-lo if lo < 0 else 0), s.step)
            for s, (lo, _, interior) in
            zip(cropped_input_slice, padding_config))

        # assert np.all(outval[output_slice] == init_value)
        outval = index_update(outval, output_slice, input[input_slice])

    return fa.Parray((outval, output_mask))


fa.update_rules[lax.pad_p] = pad_update


def conv_general_dilated_outmask(lhs_mask, rhs_mask, **params):
    # Note: we assume that rhs_mask doesn't change
    lhs_mask, rhs_mask = np.float32(lhs_mask), np.float32(rhs_mask)
    out = onp.array(lax.conv_general_dilated(lhs_mask, rhs_mask, **params))
    full_out = onp.array(lax.conv_general_dilated(
        onp.ones_like(lhs_mask), onp.ones_like(rhs_mask), **params))
    return out == full_out


def conv_general_dilated_update_slice_op(
        slc, out, lhs, rhs, window_strides, padding,
        lhs_dilation=None, rhs_dilation=None, dimension_numbers=None,
        precision=None):
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    lhs_shape = onp.take(np.shape(lhs), lhs_spec)
    rhs_shape = onp.take(np.shape(rhs), rhs_spec)
    out_slc = [slc[i] for i in out_spec]
    pad_low, _ = onp.transpose(padding)
    window_shape = lax.lax._dilate_shape(rhs_shape, rhs_dilation)[2:]
    out_start, out_stop = onp.transpose([[s.start, s.stop] for s in out_slc])
    out_start_dilated = out_start[2:] * onp.array(window_strides)
    out_stop_dilated = (out_stop[2:] - 1) * onp.array(window_strides) + 1
    lhs_bounds = out_start_dilated, out_stop_dilated + window_shape - 1
    lhs_start_dilated = out_start_dilated - pad_low
    lhs_stop_dilated = out_stop_dilated + window_shape - 1 - pad_low
    lhs_start = onp.clip((lhs_start_dilated - 1) // lhs_dilation + 1, 0,
                         lhs_shape[2:])
    lhs_stop = onp.clip((lhs_stop_dilated - 1) // lhs_dilation + 1, 0,
                        lhs_shape[2:])
    lhs_start, lhs_stop = onp.clip(
        (onp.array([lhs_start_dilated, lhs_stop_dilated]) - 1) // lhs_dilation + 1, 0,
                                   lhs_shape[2:])

    if onp.any(lhs_start == lhs_stop):
        return out

    sub_pad_low = onp.maximum(
        onp.multiply(lhs_start, lhs_dilation) - lhs_start_dilated, 0)
    sub_pad_high = onp.maximum(
        lhs_stop_dilated - onp.multiply((lhs_stop - 1), lhs_dilation) - 1, 0)
    sub_padding = zip(sub_pad_low, sub_pad_high)
    lhs_slice = ((out_slc[0], slice(None)) +
                 tuple(slice(int(s), int(e)) for s, e in
                       zip(lhs_start, lhs_stop)))
    new = lax.conv_general_dilated(
        lhs[tuple(onp.take(lhs_slice, onp.argsort(lhs_spec)))], rhs,
        window_strides=window_strides, padding=sub_padding,
        lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers, precision=precision)
    return index_update(out, slc, new)


def conv_general_dilated_update(old_out, lhs, rhs, window_strides, padding,
                                lhs_dilation, rhs_dilation, dimension_numbers,
                                feature_group_count, lhs_shape, rhs_shape,
                                precision):
    lhs, lhs_mask = lhs
    rhs, rhs_mask = rhs

    if not np.all(rhs_mask) or feature_group_count > 1:
        raise NotImplementedError

    outval, old_outmask = old_out

    outmask = conv_general_dilated_outmask(
        lhs_mask, rhs_mask, window_strides=window_strides, padding=padding,
        lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers)

    # if dimension_numbers is not None:
    #     assert dimension_numbers.lhs_spec == tuple(range(np.ndim(lhs)))
    #     assert dimension_numbers.rhs_spec == tuple(range(np.ndim(rhs)))
    #     assert dimension_numbers.out_spec == tuple(range(np.ndim(outval)))

    new_mask = outmask & ~old_outmask
    for slice in util.mask_to_slices(new_mask):
        outval = conv_general_dilated_update_slice_op(
            slice, outval, lhs, rhs, window_strides, padding,
            lhs_dilation, rhs_dilation, dimension_numbers, precision)

    return fa.Parray((outval, outmask))


fa.update_rules[lax.conv_general_dilated_p] = conv_general_dilated_update

def device_put_update(old_out, x, device_num=0):
    x, mask = util.tree_unmask(x)
    return fa.parray(xla.device_put_p.bind(x, device_num=device_num), mask)
fa.update_rules[xla.device_put_p] = device_put_update
