import numpy as onp

import jax.core as jc
from jax import vjp
import jax.numpy as np
from jax.util import curry, safe_zip, safe_map, unzip2, partial
import jax.lax as lax
import jax.interpreters.xla as xla
import jax.interpreters.ad as ad

import fastar.interpreter as fa
import fastar.primitives as primitives
import fastar.util as util

map = safe_map
zip = safe_zip

def _add_at(arr, idxs, vals):
    # This may not be efficient
    def take(arr, idxs):
        return arr[idxs]

    ans, take_vjp = vjp(lambda arr: take(arr, idxs), arr)
    assert ans.shape == vals.shape
    return arr + take_vjp(vals)[0]


@curry
def unary_ufunc_sub(func, old_out, a):
    a, a_mask = a
    outval, old_outmask = old_out[0]
    new_mask = a_mask & ~old_outmask
    if onp.all(~new_mask):
        return old_out[0]
    slices = util.mask_to_slices(new_mask)
    for slc in slices:
        assert np.all(outval[slc] == 0)
        outval = _add_at(outval, slc, func.bind(a[slc]))
    return fa.Masked((outval, a_mask))

@curry
def binary_ufunc_sub(func, old_out, a, b):
    a, a_mask = a
    b, b_mask = b
    outval, old_outmask = old_out[0]
    outmask = a_mask & b_mask
    new_mask = outmask & ~old_outmask
    if onp.all(~new_mask):
        return old_out[0]
    out_slices = util.mask_to_slices(new_mask)
    for slc in out_slices:
        a_slice = tuple(s if dim_sz > 1 else slice(None, None)
                        for s, dim_sz in zip(slc[-np.ndim(a):], np.shape(a)))
        b_slice = tuple(s if dim_sz > 1 else slice(None, None)
                        for s, dim_sz in zip(slc[-np.ndim(b):], np.shape(b)))
        assert np.all(outval[slc] == 0)
        new_outval = _add_at(outval, slc, func.bind(a[a_slice], b[b_slice]))
    return fa.Masked((new_outval, outmask))

fa.sub_rules[lax.sin_p] = unary_ufunc_sub(lax.sin_p)
fa.sub_rules[lax.add_p] = binary_ufunc_sub(lax.add_p)
fa.sub_rules[lax.sub_p] = binary_ufunc_sub(lax.sub_p)
fa.sub_rules[lax.mul_p] = binary_ufunc_sub(lax.mul_p)

# fa.Masked convolution
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
    out_stop   = onp.array([s.stop   for s in slc[2:]])
    out_start = out_start * np.array(window_strides)
    out_stop = out_stop * np.array(window_strides)
    lhs_start_dilated = onp.subtract(out_start, pad_low)
    lhs_stop_dilated   = onp.subtract(out_stop + window_shape - 1, pad_low)
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
    sub_pad_high = np.where(lhs_stop > lhs_shape_dil, lhs_stop - lhs_shape_dil, 0)
    sub_padding = zip(sub_pad_low, sub_pad_high)
    lhs_start = np.where(lhs_start > 0, lhs_start, 0)
    lhs_stop = np.where(lhs_stop > lhs_shape_dil, lhs_shape_dil, lhs_stop)
    lhs_slice = ((slc[0], slice(None, None)) +
                 tuple(slice(int(s), int(e)) for s, e in zip(lhs_start, lhs_stop)))
    new = lax.conv_general_dilated(lhs[lhs_slice], rhs, window_strides,
                                   sub_padding, lhs_dilation=lhs_dilation,
                                   rhs_dilation=rhs_dilation,
                                   dimension_numbers=dimension_numbers)
    return _add_at(out, slc, new)


def conv_general_dilated_masked_sub(
        old_out, lhs, rhs, window_strides, padding, lhs_dilation=None,
        rhs_dilation=None, dimension_numbers=None, rhs_mask=None,
        **unused_kwargs):
    lhs, lhs_msk = lhs
    rhs, rhs_var_msk = rhs
    if not np.all(rhs_var_msk):
        raise NotImplementedError
    outval, old_outmsk = old_out[0]
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
    return fa.Masked((outval, outmsk))

fa.sub_rules[primitives._conv_general_dilated_masked_p] = (
    conv_general_dilated_masked_sub)
