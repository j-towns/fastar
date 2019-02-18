import jax.lax as lax
import jax.numpy as np
import numpy as onp
from jax import vjp
from jax.util import curry, safe_zip, safe_map, unzip2

import fastar.interpreter as fa
import fastar.primitives as primitives
import fastar.util as util

map = safe_map
zip = safe_zip


def _add_at(arr, idxs, vals):
    def take(arr, idxs):
        return arr[idxs]

    ans, take_vjp = vjp(lambda arr: take(arr, idxs), arr)
    assert ans.shape == vals.shape
    return arr + take_vjp(vals)[0]


def sliceableop_update(func, old_out, *args, output_mask,
                       input_slices_from_output_slice, **params):
    """Update rule for operations where any slice of their (only) output can be
    calculated by applying the operation itself to one slice of each input.

    :param input_slices_from_output_slice:
        Specifies which input slices are required for an output slice."""

    (outval, old_outmask), = old_out
    new_mask = output_mask & ~old_outmask
    output_slices = util.mask_to_slices(new_mask)
    for output_slice in output_slices:
        assert np.all(outval[output_slice] == fa.init_value)

        input_slices = input_slices_from_output_slice(output_slice)
        sliced_inputs = tuple(arg[s] for arg, s in zip(args, input_slices))
        sliced_output = func.bind(*sliced_inputs, **params)
        outval = _add_at(outval, output_slice, sliced_output)

    return fa.Parray((outval, output_mask))


@curry
def unop_update(func, old_out, a):
    a, a_mask = a
    return sliceableop_update(func, old_out, a, output_mask=a_mask,
                              input_slices_from_output_slice=lambda s: (s,))


unops = [lax.sin_p]
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
        func, old_out, a, b, output_mask=a_mask & b_mask,
        input_slices_from_output_slice=
        lambda s: (broadcast_slice(s, a.shape), broadcast_slice(s, b.shape)))


binops = [lax.add_p, lax.sub_p, lax.mul_p]
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
        func, old_out, a,
        output_mask=onp.equal(onp.sum(a_mask.astype(int), axis=axes),
                              onp.prod([a_mask.shape[axis] for axis in axes])),
        input_slices_from_output_slice=input_slices_from_output_slice,
        axes=axes, **params)


reduce_ops = [lax.reduce_sum_p, lax.reduce_min_p, lax.reduce_max_p]
for op in reduce_ops:
    fa.update_rules[op] = reduce_update(op)


def dot_update(old_out, a, b):
    a, a_mask = a
    b, b_mask = b
    return sliceableop_update(
        lax.dot_p, old_out, a, b, output_mask=onp.equal(
            onp.dot(a_mask.astype(int), b_mask.astype(int)),
            onp.shape(b_mask)[0]),
        input_slices_from_output_slice=lambda s: (
            (s[0], slice(None)) if len(s) > 0 else (slice(None),),
            (slice(None), s[1]) if len(s) > 1 else (slice(None),)))


fa.update_rules[lax.dot_p] = dot_update


def dot_general_update(old_out, a, b, dimension_numbers):
    a, a_mask = a
    b, b_mask = b
    ((a_contracting_dims, b_contracting_dims),
     (a_batch_dims, b_batch_dims)) = dimension_numbers

    contraction_ratio = np.prod([a.shape[d] for d in a_contracting_dims])

    o = lax.dot_general(a_mask.astype(np.float64), b_mask.astype(np.float64),
                        dimension_numbers=dimension_numbers)
    output_mask = onp.equal(o, onp.full_like(o, contraction_ratio))

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
        lax.dot_general_p, old_out, a, b, output_mask=output_mask,
        input_slices_from_output_slice=input_slices_from_output_slice,
        dimension_numbers=dimension_numbers)


fa.update_rules[lax.dot_general_p] = dot_general_update


def transpose_update(old_out, a, permutation):
    a, a_mask = a

    return fa.Parray((np.transpose(a, permutation),
                      onp.transpose(a_mask, permutation)))


fa.update_rules[lax.transpose_p] = transpose_update


def reverse_update(old_out, a, dimensions):
    a, a_mask = a

    return fa.Parray((lax.rev(a, dimensions),
                      lax.rev(a_mask, dimensions)))


fa.update_rules[lax.rev_p] = reverse_update


def pad_update(old_out, a, padding_value, padding_config):
    a, a_mask = a
    padding_value, padding_value_mask = padding_value
    (outval, old_outmask), = old_out

    unpad_slice = tuple(slice(lo, -high, None if interior == 0 else interior + 1)
                        for (lo, high, interior) in padding_config)
    unpad = lambda x: x[unpad_slice]

    def pad_slices():
        for index, (lo, hi, interior) in enumerate(padding_config):
            def nonoverlapping(s):
                return unpad_slice[:index] + (s,) + \
                       tuple([slice(None)] * (old_outmask.ndim - index - 1))

            yield nonoverlapping(slice(lo))
            for i in range(interior):
                yield nonoverlapping(slice(lo + i + 1, -hi, (interior + 1)))
            yield nonoverlapping(slice(-hi, None))

    old_padding_value_mask = any(onp.any(old_outmask[s]) for s in pad_slices())
    new_padding_value_mask = padding_value_mask and not old_padding_value_mask

    output_mask = old_outmask.copy()
    output_mask[unpad_slice] = a_mask
    if new_padding_value_mask:
        for s in pad_slices():
            outval = _add_at(outval, s, np.broadcast_to(padding_value,
                                                        output_mask[s].shape))
            output_mask[s] = True

    new_input_mask = a_mask & ~unpad(old_outmask)
    input_slices = util.mask_to_slices(new_input_mask)
    for input_slice in input_slices:
        output_slice = tuple(
            slice(lo + s.start * (interior + 1), lo + s.stop * (interior + 1), interior + 1)
            for s, (lo, _, interior) in zip(input_slice, padding_config))
        assert np.all(outval[output_slice] == fa.init_value)
        outval = _add_at(outval, output_slice, a[input_slice])

    return fa.Parray((outval, output_mask))


fa.update_rules[lax.pad_p] = pad_update


# fa.Parray convolution
def conv_general_dilated_outmask(lhs_mask, rhs_mask, **params):
    # Note: we assume that rhs_mask doesn't change
    in_chan = onp.shape(lhs_mask)[1]
    assert in_chan == onp.shape(rhs_mask)[1]
    lhs_mask, rhs_mask = np.float32(lhs_mask), np.float32(rhs_mask)
    out = onp.array(
        lax.conv_general_dilated(lhs_mask, rhs_mask, **params))
    full_out = onp.array(lax.conv_general_dilated(
        onp.ones_like(lhs_mask), rhs_mask, **params))
    return out == full_out


def conv_general_dilated_masked_slice(
        slc, out, lhs, rhs, rhs_msk, window_strides, padding,
        lhs_dilation=None, rhs_dilation=None, dimension_numbers=None):
    lhs_shape, rhs_shape = np.shape(lhs), np.shape(rhs)
    pad_low, pad_high = unzip2(padding)
    window_shape = lax._dilate_shape(rhs_shape, rhs_dilation)[2:]
    lhs_shape_dil = lax._dilate_shape(lhs_shape, lhs_dilation)[2:]
    out_start = onp.array([s.start for s in slc[2:]])
    out_stop = onp.array([s.stop for s in slc[2:]])
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
    lhs_slice = ((slc[0], slice(None)) +
                 tuple(slice(int(s), int(e)) for s, e in
                       zip(lhs_start, lhs_stop)))
    new = lax.conv_general_dilated(lhs[lhs_slice], rhs, window_strides,
                                   sub_padding, lhs_dilation=lhs_dilation,
                                   rhs_dilation=rhs_dilation,
                                   dimension_numbers=dimension_numbers)
    return _add_at(out, slc, new)


def conv_general_dilated_masked_update(
        old_out, lhs, rhs, window_strides, padding, lhs_dilation=None,
        rhs_dilation=None, dimension_numbers=None, rhs_mask=None,
        **unused_kwargs):
    lhs, lhs_msk = lhs
    rhs, rhs_var_msk = rhs
    if not np.all(rhs_var_msk):
        raise NotImplementedError
    (outval, old_outmsk), = old_out if old_out else fa._init_output(
        lax.conv_general_dilated_p, lhs, rhs,
        window_strides=tuple(window_strides), padding=tuple(padding),
        lhs_dilation=tuple(lhs_dilation), rhs_dilation=tuple(rhs_dilation),
        dimension_numbers=dimension_numbers, rhs_mask=rhs_mask,
        lhs_shape=lhs.shape, rhs_shape=rhs.shape)
    if dimension_numbers is not None:
        assert dimension_numbers.lhs_spec == tuple(range(np.ndim(lhs)))
        assert dimension_numbers.rhs_spec == tuple(range(np.ndim(rhs)))
        assert dimension_numbers.out_spec == tuple(range(np.ndim(outval)))
    outmsk = conv_general_dilated_outmask(
        lhs_msk, rhs_mask, window_strides=window_strides, padding=padding,
        lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
        dimension_numbers=dimension_numbers)

    new_msk = outmsk & ~old_outmsk
    if np.all(~new_msk):
        return old_out[0]
    for slc in util.mask_to_slices(new_msk):
        outval = conv_general_dilated_masked_slice(
            slc, outval, lhs, rhs, rhs_mask, window_strides, padding,
            lhs_dilation, rhs_dilation, dimension_numbers)
    return fa.Parray((outval, outmsk))


fa.update_rules[primitives._conv_general_dilated_masked_p] = (
    conv_general_dilated_masked_update)
