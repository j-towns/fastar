import numpy as onp
from jax import vjp
from jax.api_util import (pytree_fun_to_jaxtupletree_fun,
                          pytree_to_jaxtupletree, wraps)
import jax.core as jc
import jax.interpreters.xla as xla
import jax.interpreters.partial_eval as pe
import jax.linear_util as lu
import jax.numpy as np
import jax.lax as lax
from jax.util import safe_map, partial, unzip2

map = safe_map
setminus = lambda a, b: a & ~b

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
        map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar), env


def fastpass(jaxpr, consts, old_env, *args):
    def read(v):
        val = env[v]
        assert isinstance(val, Masked)
        return env[v]

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
    write_const(jc.unitvar, jc.unit)
    map(write_const, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        _, old_inmasks = unzip2(map(read_old, eqn.invars))
        old_outvals, old_outmasks = unzip2(map(read_old, eqn.outvars))
        in_vals, in_masks = unzip2(map(read, eqn.invars))
        new_outmask = outmask_rules[eqn.primitive](*in_masks, **eqn.params)
        new_outmasks = list(new_outmask) if eqn.destructure else [new_outmask]
        to_compute = map(setminus, new_outmasks, old_outmasks)
        if eqn.bound_subjaxprs:
            raise NotImplementedError
        if np.any(to_compute):
            to_compute = map(rectangular_mask_to_slice, to_compute)
            new = get_subprimitive(eqn.primitive)(to_compute, *in_vals,
                                                  **eqn.params)
            new = list(new) if eqn.destructure else [new]
            ans = map(add_at, old_outvals, to_compute, new)
            outvals = map(Masked, zip(ans, new_outmasks))
        else:
            outvals = map(Masked, zip(old_outvals, old_outmasks))
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
    return arr + take_vjp(vals)[0]

def new_mask(old_mask, in_mask):
    return in_mask & ~old_mask

def binary_ufunc_outmask(a_mask, b_mask):
    return a_mask & b_mask

def unary_ufunc_outmask(a_mask):
    return a_mask

outmask_rules = {}

outmask_rules[lax.add_p] = binary_ufunc_outmask
outmask_rules[lax.sub_p] = binary_ufunc_outmask
outmask_rules[lax.mul_p] = binary_ufunc_outmask
outmask_rules[lax.sin_p] = unary_ufunc_outmask

sub_rules = {}

def get_subprimitive(p):
    try:
        return sub_rules[p]
    except KeyError:
        raise NotImplementedError(
            "Masked computation rule for '{}' not implemented".format(p))

def binary_ufunc_sub(func, to_comp, a, b):
    to_comp = to_comp[0]
    a_slice = tuple(s if dim_sz > 1 else slice(None, None)
                    for s, dim_sz in zip(to_comp[-np.ndim(a):], np.shape(a)))
    b_slice = tuple(s if dim_sz > 1 else slice(None, None)
                    for s, dim_sz in zip(to_comp[-np.ndim(b):], np.shape(b)))
    return func.bind(a[a_slice], b[b_slice])

def unary_ufunc_sub(func, to_comp, a):
    return func.bind(a[to_comp[0]])


sub_rules[lax.add_p] = partial(binary_ufunc_sub, lax.add_p)
sub_rules[lax.sub_p] = partial(binary_ufunc_sub, lax.sub_p)
sub_rules[lax.mul_p] = partial(binary_ufunc_sub, lax.mul_p)
sub_rules[lax.sin_p] = partial(unary_ufunc_sub, lax.sin_p)

def rectangular_mask_to_slice(mask):
    nonzero_idxs = onp.argwhere(mask)
    starts = onp.amin(nonzero_idxs, axis=0)
    ends = onp.amax(nonzero_idxs, axis=0)
    return tuple(slice(s, e + 1) for s, e in zip(starts, ends))


if __name__ is "__main__":
    f = lambda x: np.sin(2. * x + 1.)
    jaxpr, consts = make_jaxpr(f)(np.array([0., 0.]))
    _, env = firstpass(jaxpr, consts, np.array([0., 0.]))
    val, env_ = fastpass(jaxpr, consts, env,
                         Masked((np.array([0., 0]), onp.array([False, False]))))
    val, env__ = fastpass(jaxpr, consts, env_,
                          Masked((np.array([1., 0]), onp.array([True, True]))))
    val, env___ = fastpass(jaxpr, consts, env__,
                           Masked((np.array([1., 2]), onp.array([True, True]))))
