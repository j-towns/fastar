import numpy as onp
import jax.numpy as np
import jax.lax as lax
from jax.util import safe_map, partial, unzip2

map = safe_map
setminus = lambda a, b: a & ~b

def firstpass(jaxpr, consts, freevar_vals, *args):
    # Similar to core.eval_jaxpr but returns env
    def read(v):
        return env[v]

    def write(v, val):
        env[v] = val

    env = {}
    write(unitvar, unit)
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    map(write, jaxpr.freevars, freevar_vals)
    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.invars)
        if eqn.bound_subjaxprs:
            raise NotImplementedError
        ans = eqn.primitive.bind(*(subfuns + in_vals), **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar), env


def fastpass(jaxpr, consts, freevar_vals, old_env, *args):
    def read(v):
        return env[v]

    def read_old(v):
        return old_env[v]

    def write(v, val):
        if not isinstance(val, Masked):
            val = Masked(val, default_mask(val))
        env[v] = val

    env = {}
    write(unitvar, unit)
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    map(write, jaxpr.freevars, freevar_vals)
    for eqn in jaxpr.eqns:
        _, old_inmasks = unzip2(map(read_old, eqn.invars))
        old_outvals, old_outmasks = unzip2(map, read_old, eqn.outvars)
        in_vals, in_masks = unzip2(map(read, eqn.invars))
        new_outmask = outmask_rules[eqn.primitive](*in_masks, **eqn.params)
        new_outmasks = list(new_outmask) if eqn.destructure else [new_outmask]
        to_compute = map(setminus, new_outmasks, old_outmasks)
        if eqn.bound_subjaxprs:
            raise NotImplementedError
        new = get_subprimitive(eqn.primitive)(to_compute, *in_vals, **eqn.params)
        ans = map(add_at, old_outvals, to_compute, new)
        outvals = zip(list(ans) if eqn.destructure else [ans], new_outmasks)
        map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar), env

class Masked(tuple):
    pass

def default_mask(val):
    return onp.full_like(val, True, dtype=bool)

def new_mask(old_mask, in_mask):
    return in_mask & ~old_mask

def binary_ufunc_outmask(a_mask, b_mask):
    return a_mask & b_mask

outmask_rules = {}

outmask_rules[lax.add_p] = binary_ufunc_outmask
outmask_rules[lax.sub_p] = binary_ufunc_outmask
outmask_rules[lax.mul_p] = binary_ufunc_outmask

sub_rules = {}

def get_subprimitive(p):
  try:
    return sub_rules[p]
  except KeyError:
    raise NotImplementedError(
      "Masked computation rule for '{}' not implemented".format(p))

def binary_ufunc_sub(func, to_comp, a, b):
    to_comp = rectangular_mask_to_slice(to_comp)
    a_slice = tuple(s if dim_sz > 1 else slice(None, None)
                    for s, dim_sz in zip(to_comp[-np.ndim(a):], np.shape(a)))
    b_slice = tuple(s if dim_sz > 1 else slice(None, None)
                    for s, dim_sz in zip(to_comp[-np.ndim(b):], np.shape(b)))
    return func(a[a_slice], b[b_slice])

outmask_rules[lax.add_p] = partial(binary_ufunc_sub, lax.add_p)
outmask_rules[lax.sub_p] = partial(binary_ufunc_sub, lax.sub_p)
outmask_rules[lax.mul_p] = partial(binary_ufunc_sub, lax.mul_p)

def rectangular_mask_to_slice(mask):
    nonzero_idxs = onp.argwhere(mask)
    starts = onp.amin(nonzero_idxs, axis=0)
    ends = onp.amax(nonzero_idxs, axis=0)
    return tuple(slice(s, e + 1) for s, e in zip(starts, ends))
