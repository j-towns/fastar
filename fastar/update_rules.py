import jax.lax as lax
import jax.numpy as np
import numpy as onp
from jax import vjp
from jax.abstract_arrays import make_shaped_array
from jax.util import curry, safe_zip, safe_map, unzip2

import fastar.util as util
from . import interpreter as fa

map = safe_map
zip = safe_zip

init_value = 0


def _init_output(func, *args, **params):
    # TODO: generalize to jaxvals
    args = map(make_shaped_array, args)
    abstract_out = func.abstract_eval(*args, **params)
    outval = lax.full(abstract_out.shape, init_value, abstract_out.dtype)
    outmask = onp.zeros(abstract_out.shape, bool)
    return (outval, outmask),


def _add_at(arr, idxs, vals):
    def take(arr, idxs):
        return arr[idxs]

    ans, take_vjp = vjp(lambda arr: take(arr, idxs), arr)
    assert ans.shape == vals.shape
    return arr + take_vjp(vals)[0]


def sliceableop_update(func, old_out, output_mask,
                       input_slices_from_output_slice, *args, **params):
    """Update rule for operations where any slice of their (only) output can be
    calculated by applying the operation itself to one slice of each input.

    :param input_slices_from_output_slice:
        Specifies which input slices are required for an output slice."""

    (outval, old_outmask), = _init_output(func, *args, **params) if \
        old_out is None else old_out

    new_mask = output_mask & ~old_outmask
    output_slices = util.mask_to_slices(new_mask)
    for output_slice in output_slices:
        assert np.all(outval[output_slice] == init_value)

        input_slices = input_slices_from_output_slice(output_slice)
        sliced_inputs = tuple(arg[s] for arg, s in zip(args, input_slices))
        sliced_output = func.bind(*sliced_inputs, **params)
        outval = _add_at(outval, output_slice, sliced_output)

    return fa.Parray((outval, output_mask))


@curry
def unop_update(func, old_out, a):
    a, a_mask = a
    return sliceableop_update(func, old_out, a_mask, lambda s: (s,), a)


unops = [lax.abs_p, lax.ceil_p, lax.cos_p, lax.sin_p, lax.exp_p,
         lax.floor_p, lax.log_p, lax.neg_p, lax.sign_p, lax.tanh_p]
for op in unops:
    fa.update_rules[op] = unop_update(op)


@curry
def binop_update(func, old_out, a, b):
    a, a_mask = a
    b, b_mask = b

    def broadcast_slice(s, shape):
        return tuple(s if dim_sz > 1 else slice(None)
                     for s, dim_sz in zip(s[-len(shape):], shape))

    return sliceableop_update(
        func, old_out, a_mask & b_mask,
        lambda s: (broadcast_slice(s, a.shape), broadcast_slice(s, b.shape)),
        a, b)


binops = [lax.add_p, lax.sub_p, lax.mul_p, lax.div_p,
          lax.rem_p, lax.max_p, lax.min_p]
for op in binops:
    fa.update_rules[op] = binop_update(op)


@curry
def reduce_update(func, old_out, a, axes, **params):
    a, a_mask = a

    def input_slices_from_output_slice(output_slice):
        a_slice = list(output_slice)
        for axis in axes:
            a_slice.insert(axis, slice(None))
        return tuple(a_slice),

    return sliceableop_update(
        func, old_out, onp.equal(onp.sum(a_mask.astype(int), axis=axes),
                                 onp.prod([a_mask.shape[axis]
                                           for axis in axes])),
        input_slices_from_output_slice, a, axes=axes, **params)


reduce_ops = [lax.reduce_sum_p, lax.reduce_min_p, lax.reduce_max_p]
for op in reduce_ops:
    fa.update_rules[op] = reduce_update(op)


def dot_update(old_out, a, b):
    a, a_mask = a
    b, b_mask = b
    return sliceableop_update(
        lax.dot_p, old_out, onp.equal(
            onp.dot(a_mask.astype(int), b_mask.astype(int)),
            onp.shape(b_mask)[0]),
        lambda s: ((s[0], slice(None)) if len(s) > 0 else (slice(None),),
                   (slice(None), s[1]) if len(s) > 1 else (slice(None),)),
        a, b)


fa.update_rules[lax.dot_p] = dot_update


def dot_general_update(old_out, a, b, dimension_numbers):
    a, a_mask = a
    b, b_mask = b
    ((a_contracting_dims, b_contracting_dims),
     (a_batch_dims, b_batch_dims)) = dimension_numbers

    contraction_ratio = np.prod([a.shape[d] for d in a_contracting_dims])

    o = lax.dot_general(a_mask.astype(np.float64), b_mask.astype(np.float64),
                        dimension_numbers=dimension_numbers)
    output_mask = onp.equal(onp.array(o), onp.full_like(o, contraction_ratio))

    def input_slices_from_output_slice(output_slice):
        def result(ndim, contracting_dims, batch_dims):
            other_dims = [i for i in range(ndim) if i not in
                          contracting_dims and i not in batch_dims]

            return tuple(slice(None) if i in contracting_dims else
                         (output_slice[batch_dims[list(batch_dims).index(i)]]
                          if i in batch_dims else
                          output_slice[other_dims[other_dims.index(i)]])
                         for i in range(ndim))

        return (result(a.ndim, a_contracting_dims, a_batch_dims),
                result(b.ndim, b_contracting_dims, b_batch_dims))

    return sliceableop_update(
        lax.dot_general_p, old_out, output_mask,
        input_slices_from_output_slice, a, b,
        dimension_numbers=dimension_numbers)


fa.update_rules[lax.dot_general_p] = dot_general_update


@curry
def rearrange_update(func, old_out, *args, **params):
    args, args_mask = zip(*args)

    return fa.Parray((func.bind(*args, **params),
                      onp.array(func.bind(*args_mask, **params))))


rearrange_ops = [lax.reshape_p, lax.transpose_p, lax.rev_p, lax.concatenate_p]

for op in rearrange_ops:
    fa.update_rules[op] = rearrange_update(op)


def pad_update(old_out, input, padding_value, padding_config):
    for (lo, hi, interior) in padding_config:
        if interior > 0 and (lo < 0 or hi < 0): raise NotImplementedError(
            "Interior and negative padding on same axis not yet implemented.")

    input, input_mask = input
    padding_value, padding_value_mask = padding_value
    (outval, old_outmask), = _init_output(lax.pad_p, input, padding_value,
                                          padding_config=padding_config) if \
        old_out is None else old_out

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
            outval = _add_at(outval, s, np.broadcast_to(padding_value,
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

        assert np.all(outval[output_slice] == init_value)
        outval = _add_at(outval, output_slice, input[input_slice])

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
        lhs_dilation=None, rhs_dilation=None, dimension_numbers=None):
    permute = lambda it, perm: [it[i] for i in perm]
    unpermute = lambda it, perm: [it[i] for i in onp.argsort(perm)]
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    lhs_shape = permute(np.shape(lhs), lhs_spec)
    rhs_shape = permute(np.shape(rhs), rhs_spec)
    out_slc = permute(slc, out_spec)
    pad_low, pad_high = unzip2(padding)
    window_shape = lax._dilate_shape(rhs_shape, rhs_dilation)[2:]
    lhs_shape_dil = lax._dilate_shape(lhs_shape, lhs_dilation)[2:]
    out_start = onp.array([s.start for s in out_slc[2:]])
    out_stop =  onp.array([s.stop  for s in out_slc[2:]])
    out_start = out_start * np.array(window_strides)
    out_stop = out_stop * np.array(window_strides)
    lhs_start_dilated = onp.subtract(out_start, pad_low)
    lhs_stop_dilated = onp.subtract(out_stop + window_shape - 1, pad_low)
    lhs_start = np.where(
        lhs_start_dilated > 0,
        np.where(lhs_start_dilated < lhs_shape_dil,
                 lhs_start_dilated // lhs_dilation,
                 lhs_start_dilated - (lhs_shape_dil - lhs_shape[2:])),
        lhs_start_dilated // lhs_dilation)
    lhs_stop = np.where(
        lhs_stop_dilated > 0,
        np.where(lhs_stop_dilated < lhs_shape_dil,
                 lhs_stop_dilated // lhs_dilation,
                 lhs_stop_dilated - (lhs_shape_dil - lhs_shape[2:])),
        lhs_stop_dilated // lhs_dilation)
    sub_pad_low = np.where(lhs_start < 0, -lhs_start, 0)
    sub_pad_high = np.where(lhs_stop > lhs_shape_dil, lhs_stop - lhs_shape_dil,
                            0)
    sub_padding = zip(sub_pad_low, sub_pad_high)
    lhs_start = np.where(lhs_start > 0, lhs_start, 0)
    lhs_stop = np.where(lhs_stop > lhs_shape_dil, lhs_shape_dil, lhs_stop)
    lhs_slice = ((out_slc[0], slice(None)) +
                 tuple(slice(int(s), int(e)) for s, e in
                       zip(lhs_start, lhs_stop)))
    new = lax.conv_general_dilated(
        lhs[unpermute(lhs_slice, lhs_spec)], rhs,
        window_strides=window_strides, padding=sub_padding,
        lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers)
    return _add_at(out, slc, new)


def conv_general_dilated_update(old_out, lhs, rhs, window_strides, padding,
                                lhs_dilation, rhs_dilation, dimension_numbers,
                                lhs_shape, rhs_shape):
    lhs, lhs_mask = lhs
    rhs, rhs_mask = rhs

    if not np.all(rhs_mask):
        raise NotImplementedError

    (outval, old_outmask), = old_out if old_out else _init_output(
        lax.conv_general_dilated_p, lhs, rhs,
        window_strides=tuple(window_strides), padding=tuple(padding),
        lhs_dilation=tuple(lhs_dilation), rhs_dilation=tuple(rhs_dilation),
        dimension_numbers=dimension_numbers,
        lhs_shape=lhs_shape, rhs_shape=rhs_shape)

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
            lhs_dilation, rhs_dilation, dimension_numbers)

    return fa.Parray((outval, outmask))


fa.update_rules[lax.conv_general_dilated_p] = conv_general_dilated_update
