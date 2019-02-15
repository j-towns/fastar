import jax.lax as lax
import jax.numpy as np
import numpy as onp
from jax import vjp
from jax.abstract_arrays import make_shaped_array
from jax.util import curry, safe_zip, safe_map, unzip2

import fastar.interpreter as fa
import fastar.primitives as primitives
import fastar.util as util

map = safe_map
zip = safe_zip
noneslice = slice(None, None)


def _add_at(arr, idxs, vals):
    def take(arr, idxs):
        return arr[idxs]

    ans, take_vjp = vjp(lambda arr: take(arr, idxs), arr)
    assert ans.shape == vals.shape
    return arr + take_vjp(vals)[0]


def _init_output(func, *args, init_value=0, **params):
    # TODO: generalize to jaxvals
    args = map(make_shaped_array, args)
    abstract_out = func.abstract_eval(*args, **params)
    outval = lax.full(abstract_out.shape, init_value, abstract_out.dtype)
    outmask = onp.zeros(abstract_out.shape, bool)
    return outval, outmask


def sliceableop_update(func, old_out, *args, output_mask,
                       input_slices_from_output_slice,
                       init_output=_init_output,
                       sliced_output_from_sliced_inputs=None, init_value=0,
                       **params):
    """Applicable to all operations with the following property:
    Any slice of the output can be calculated from one slice of each input.
    sliced_output_from_sliced_inputs determines which input slices are used.
    By default, input slices are combined by the operator 'func'.
    This behavior can be overriden with input_slices_from_output_slice."""

    if sliced_output_from_sliced_inputs is None:
        sliced_output_from_sliced_inputs = lambda sliced_inputs: func.bind(
            *sliced_inputs, **params)

    outval, old_outmask = old_out[0] if old_out else init_output(
        func, *args, init_value=init_value, **params)
    new_mask = output_mask & ~old_outmask
    output_slices = util.mask_to_slices(new_mask)
    for output_slice in output_slices:
        input_slices = input_slices_from_output_slice(output_slice)
        sliced_inputs = list(
            arg[input_slice] for arg, input_slice in zip(args, input_slices))
        sliced_output = sliced_output_from_sliced_inputs(sliced_inputs)

        assert np.all(outval[output_slice] == init_value)
        outval = _add_at(outval, output_slice, sliced_output)

    return fa.Parray((outval, output_mask))


@curry
def unop_update(func, old_out, a):
    a, a_mask = a
    return sliceableop_update(func, old_out, a, output_mask=a_mask,
                              input_slices_from_output_slice=lambda
                                  output_slice: (output_slice,))


@curry
def reduce_update(func, old_out, a, axes, **params):
    a, a_mask = a

    def input_slices_from_output_slice(output_slice):
        a_slice = list(output_slice)
        for axis in axes:
            a_slice.insert(axis, noneslice)
        return tuple(a_slice),

    return sliceableop_update(
        func, old_out, a,
        output_mask=onp.equal(onp.sum(a_mask.astype(int), axis=axes),
                              onp.prod([a_mask.shape[axis] for axis in axes])),
        input_slices_from_output_slice=input_slices_from_output_slice,
        axes=axes, **params)


@curry
def binop_update(func, old_out, a, b):
    a, a_mask = a
    b, b_mask = b
    return sliceableop_update(
        func, old_out, a, b, output_mask=a_mask & b_mask,
        input_slices_from_output_slice=lambda output_slice: (
            tuple(s if dim_sz > 1 else noneslice for s, dim_sz in
                  zip(output_slice[-np.ndim(a):], np.shape(a))),
            tuple(s if dim_sz > 1 else noneslice for s, dim_sz in
                  zip(output_slice[-np.ndim(b):], np.shape(b)))))


def dot_update(old_out, a, b):
    a, a_mask = a
    b, b_mask = b
    return sliceableop_update(
        lax.dot_p, old_out, a, b, output_mask=onp.equal(
            onp.dot(a_mask.astype(int), b_mask.astype(int)),
            onp.shape(b_mask)[0]),
        input_slices_from_output_slice=lambda s: (
            (s[0], noneslice) if len(s) > 0 else (noneslice,),
            (noneslice, s[1]) if len(s) > 1 else (noneslice,)))


unops = [lax.sin_p]
for op in unops:
    fa.update_rules[op] = unop_update(op)

reduce_ops = [lax.reduce_sum_p, lax.reduce_min_p, lax.reduce_max_p]
for op in reduce_ops:
    fa.update_rules[op] = reduce_update(op)

binops = [lax.add_p, lax.sub_p, lax.mul_p]
for op in binops:
    fa.update_rules[op] = binop_update(op)

fa.update_rules[lax.dot_p] = dot_update


def pad_update(old_out, a, padding_value, padding_config):
    a, a_mask = a
    # TODO Support unitialized padding_value.
    # Is currently assumed to be initialized from the start, mask is ignored:
    padding_value, _ = padding_value

    def init_output(func, *args, **params):
        outval, outmask = _init_output(func, *args, **params)

        for index, (lo, hi, interior) in enumerate(padding_config):
            # TODO:
            if interior != 0: raise NotImplementedError(
                "Interior padding not yet supported.")

            lower_slice = [slice(None, lo) if index == i else noneslice
                           for i, s in enumerate(outmask.shape)]
            higher_slice = [slice(-hi, None) if index == i else noneslice
                            for i, s in enumerate(outmask.shape)]

            outmask[lower_slice] = True
            outmask[higher_slice] = True

        return outval, outmask

    output_mask = onp.pad(a_mask, [(lo, hi) for lo, hi, _ in padding_config],
                          'constant', constant_values=True)

    def input_slices_from_output_slice(output_slice):
        return (tuple(slice(s.start - lo, s.stop - lo, None)
                      for s, (lo, _, _) in zip(output_slice, padding_config)),
                ())

    return sliceableop_update(
        lax.pad_p, old_out, a, padding_value, output_mask=output_mask,
        init_output=init_output, init_value=padding_value,
        input_slices_from_output_slice=input_slices_from_output_slice,
        sliced_output_from_sliced_inputs=lambda s: s[0],
        padding_config=padding_config)


fa.update_rules[lax.pad_p] = pad_update


def transpose_update(old_out, a, permutation):
    a, a_mask = a

    def inversely_permute_slice(slice):
        s = [noneslice] * len(slice)
        for i, d in enumerate(permutation):
            s[d] = slice[i]
        return s

    return sliceableop_update(
        lax.transpose_p, old_out, a,
        output_mask=onp.transpose(a_mask, permutation),
        input_slices_from_output_slice=lambda s: (inversely_permute_slice(s),),
        permutation=permutation)


fa.update_rules[lax.transpose_p] = transpose_update


def reverse_update(old_out, a, dimensions):
    a, a_mask = a

    def reverse_slice(s):
        return [slice(-s.stop if s.stop else None,
                      -s.start if s.start else None, None)
                if i in dimensions else s
                for i, s in enumerate(s)]

    return sliceableop_update(
        lax.rev_p, old_out, a,
        output_mask=onp.flip(a_mask, dimensions),
        input_slices_from_output_slice=lambda s: (reverse_slice(s),),
        dimensions=dimensions)


fa.update_rules[lax.rev_p] = reverse_update


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
    lhs_slice = ((slc[0], slice(None, None)) +
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
    outval, old_outmsk = old_out[0] if old_out else _init_output(
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
