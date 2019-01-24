from itertools import product

import numpy as onp

from jax import vjp
from jax.ad_util import zeros_like_jaxval
from jax.api_util import (pytree_fun_to_jaxtupletree_fun,
                          pytree_to_jaxtupletree, wraps)
import jax.core as jc
import jax.interpreters.ad as ad
import jax.interpreters.xla as xla
import jax.interpreters.partial_eval as pe
import jax.linear_util as lu
import jax.numpy as np
import jax.lax as lax
from jax.util import curry, partial, safe_map, unzip2

import util

map = safe_map

## Core
def make_jaxpr(f):
    def pv_like(x):
        aval = xla.abstractify(x)
        return pe.PartialVal((aval, jc.unit))

    fun = lu.wrap_init(f)

    @wraps(f)
    def jaxpr_maker(*args, **kwargs):
        jax_args, in_trees = unzip2(map(pytree_to_jaxtupletree, args))
        jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(fun, in_trees)
        pvals = map(pv_like, jax_args)
        jaxpr, _, consts = pe.trace_to_jaxpr(jaxtree_fun, pvals, **kwargs)
        return jaxpr, consts

    jaxpr_maker.__name__ = "make_jaxpr({})".format(jaxpr_maker.__name__)
    return jaxpr_maker

# Populate the cache
def firstpass(jaxpr, consts, *args):
    # Similar to jax.core.eval_jaxpr but returns env
    def read(v):
        return env[v]

    def write(v, val):
        env[v] = val

    env = {}
    write(jc.unitvar, jc.unit)
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.invars)
        if eqn.bound_subjaxprs:
            raise NotImplementedError
        ans = eqn.primitive.bind(*in_vals, **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, map(zeros_like_jaxval, outvals))
    return read(jaxpr.outvar), env


def fastpass(jaxpr, consts, old_env, *args):
    def read(v):
        val = env[v]
        assert isinstance(val, Masked)
        return val

    def read_old(v):
        val = old_env[v]
        if not isinstance(val, Masked):
            val = Masked((val, false_mask(val)))
        return val

    def write_const(v, val):
        val = Masked((val, true_mask(val)))
        env[v] = val

    def write(v, val):
        assert isinstance(val, Masked)
        env[v] = val

    env = {}
    write_const(jc.unitvar, jc.unit)          # TODO: don't need these every
    map(write_const, jaxpr.constvars, consts) # time
    map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        old_outvals = map(read_old, eqn.outvars)
        in_vals = map(read, eqn.invars)
        if eqn.bound_subjaxprs:
            raise NotImplementedError
        ans = get_subprimitive(eqn.primitive)(old_outvals, *in_vals,
                                              **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar), env

class Masked(tuple):
    pass

def true_mask(val):
    return onp.full_like(val, True, dtype=bool)

def false_mask(val):
    return onp.full_like(val, False, dtype=bool)

def add_at(arr, idxs, vals):
    # This may not be efficient
    def take(arr, idxs):
        return arr[idxs]

    ans, take_vjp = vjp(lambda arr: take(arr, idxs), arr)
    assert ans.shape == vals.shape
    return arr + take_vjp(vals)[0]


## Subcomputation rules
sub_rules = {}

def get_subprimitive(p):
    try:
        return sub_rules[p]
    except KeyError:
        raise NotImplementedError(
            "Masked computation rule for '{}' not implemented".format(p))

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
        outval = add_at(outval, slc, func.bind(a[slc]))
    return Masked((outval, a_mask))

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
        assert np.all(old_outval[new_idxs] == 0)
        new_outval = add_at(outval, new_idxs, func.bind(a[a_slice], b[b_slice]))
    return Masked((new_outval, outmask))

sub_rules[lax.sin_p] = unary_ufunc_sub(lax.sin_p)
sub_rules[lax.add_p] = binary_ufunc_sub(lax.add_p)
sub_rules[lax.sub_p] = binary_ufunc_sub(lax.sub_p)
sub_rules[lax.mul_p] = binary_ufunc_sub(lax.mul_p)

# Masked convolution
def pop_msk(fun):
    def fun_(*args, **kwargs):
        kwargs.pop('rhs_mask')
        return fun(*args, **kwargs)
    return fun_

# Semantically this primitive is no different to lax.conv_general_dilated. We
# need it so that we can correctly propagate Masked values.
conv_general_dilated_masked_p = jc.Primitive('conv_general_dilated_masked')
conv_general_dilated_masked_p.def_impl(partial(xla.apply_primitive,
                                               conv_general_dilated_masked_p))
conv_general_dilated_masked_p.def_abstract_eval(pop_msk(
    partial(lax.standard_abstract_eval, lax.conv_general_dilated_shape_rule,
            lax.conv_general_dilated_dtype_rule)))
xla.translations[conv_general_dilated_masked_p] = (
    pop_msk(lax.conv_general_dilated_translation_rule))
ad.defbilinear(conv_general_dilated_masked_p,
               pop_msk(lax.conv_general_dilated_transpose_lhs),
               pop_msk(lax.conv_general_dilated_transpose_rhs))

def conv_general_dilated_masked(
        lhs, rhs, window_strides, padding, lhs_dilation=None,
        rhs_dilation=None, dimension_numbers=None, rhs_mask=None):
  rhs_mask = util.to_tuple_tree(true_mask(rhs)
                                if rhs_mask is None else rhs_mask)
  if type(dimension_numbers) is not lax.ConvDimensionNumbers:
    dimension_numbers = lax.conv_dimension_numbers(
        lhs.shape, rhs.shape, dimension_numbers)
  if isinstance(padding, str):
    lhs_perm, rhs_perm, _ = dimension_numbers
    padding = lax.padtype_to_pads(
        onp.take(lhs.shape, lhs_perm)[2:], onp.take(rhs.shape, rhs_perm)[2:],
        window_strides, padding)
  if lhs_dilation is None:
    lhs_dilation = (1,) * (lhs.ndim - 2)
  if rhs_dilation is None:
    rhs_dilation = (1,) * (rhs.ndim - 2)
  return conv_general_dilated_masked_p.bind(
      lhs, rhs, window_strides=tuple(window_strides),
      padding=tuple(padding), lhs_dilation=tuple(lhs_dilation),
      rhs_dilation=tuple(rhs_dilation), dimension_numbers=dimension_numbers,
      rhs_mask=rhs_mask, lhs_shape=lhs.shape, rhs_shape=rhs.shape)

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
    out_start_dilated = out_start * np.array(window_strides)
    out_stop_dilated   = out_stop   * np.array(window_strides)
    lhs_start_dilated = out_start - pad_low
    lhs_stop_dilated   = out_stop + window_shape - 1 - pad_low
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
    return add_at(out, slc, new)


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
    return Masked((outval, outmsk))

sub_rules[conv_general_dilated_masked_p] = conv_general_dilated_masked_sub


if __name__ is "__main__":
    window_strides = (1, 1)
    lhs = np.reshape(np.arange(25, dtype=float), (1, 1, 5, 5))
    lhs_mask = false_mask(lhs)
    rhs = np.array([[[[1., 2., 3.],
                      [4., 0., 0.],
                      [0., 0., 0.]]]])
    rhs_mask = util.to_tuple_tree(np.bool_(rhs))
    f = lambda lhs, rhs: conv_general_dilated_masked(
        lhs, rhs, window_strides, padding='SAME', rhs_mask=rhs_mask)
    jaxpr, consts = make_jaxpr(f)(lhs, rhs)
    _, env = firstpass(jaxpr, consts, lhs, rhs)
    val, env_ = fastpass(jaxpr, consts, env, Masked((lhs, lhs_mask)),
                         Masked((rhs, true_mask(rhs))))
    lhs_mask = onp.copy(lhs_mask)
    lhs_mask[..., 0, 0] = True
    val, env__ = fastpass(jaxpr, consts, env_, Masked((lhs, lhs_mask)),
                         Masked((rhs, true_mask(rhs))))
    lhs_mask = onp.copy(lhs_mask)
    lhs_mask[..., 0, 1] = True
    val, env___ = fastpass(jaxpr, consts, env__, Masked((lhs, lhs_mask)),
                           Masked((rhs, true_mask(rhs))))
