from contextlib import contextmanager

from jax.api_util import flatten_fun_nokwargs
from jax.tree_util import (
    tree_flatten, tree_unflatten, tree_map, Partial, register_pytree_node)
import jax.core as jc
import jax.interpreters.xla as xla
from jax.ad_util import zeros_like_aval
import jax.interpreters.partial_eval as pe
import jax.linear_util as lu
from jax.util import safe_map, safe_zip, partial, unzip3, cache
from jax import abstract_arrays
from jax import jit as jit_

import fastar.util as util
from fastar.util import true_mask, false_mask

map = safe_map
zip = safe_zip

def get_aval(x):
    return abstract_arrays.raise_to_shaped(jc.get_aval(x))

## Core
init_rules = {}
def init_ans(prim, *args, **params):
    if prim in init_rules:
        return init_rules[prim](*args, **params)
    else:
        args = [get_aval(arg) for arg, _ in args]
        abstract_out = prim.abstract_eval(*args, **params)
        if prim.multiple_results:
            return [parray(zeros_like_aval(o), false_mask(o))
                    for o in abstract_out]
        else:
            return parray(zeros_like_aval(abstract_out),
                          false_mask(abstract_out))

update_rules = {}
def get_update(p):
    try:
        return update_rules[p]
    except KeyError:
        raise NotImplementedError(
            "Fastar update rule for '{}' not implemented".format(p))

# Populate the cache
def firstpass(jaxpr, consts, freevar_vals, args):
    def read(v):
        if type(v) is jc.Literal:
            return parray(v.val, true_mask(v.val))
        else:
            val = env[repr(v)]
            assert isinstance(val, Parray)
            return val

    def write(v, val):
        assert isinstance(val, Parray)
        env[repr(v)] = val

    def delete(v):
        del env[repr(v)]

    def write_subenvs(vs, env):
        subenvs[repr(vs)] = env

    env = {}
    subenvs = {}

    write(jc.unitvar, parray(jc.unit, true_mask(jc.unit)))
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    map(write, jaxpr.freevars, freevar_vals)
    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.invars)
        if eqn.bound_subjaxprs:
            subjaxprs, sub_consts, sub_freevar_vals = unzip3([(
                subjaxpr,
                map(read, const_vars),
                map(read, bound_vars))
                for subjaxpr, const_vars, bound_vars
                in eqn.bound_subjaxprs])
            ans, subenvs_ = init_ans(eqn.primitive, eqn.params, subjaxprs,
                                     sub_consts, sub_freevar_vals, in_vals)
            ans, subenvs_ = get_update(eqn.primitive)(
                eqn.params, subjaxprs, sub_consts, sub_freevar_vals, in_vals,
                subenvs_)

            write_subenvs(tuple(eqn.outvars), subenvs_)
        else:
            ans = init_ans(eqn.primitive, *in_vals, **eqn.params)
            ans = get_update(eqn.primitive)(ans, *in_vals, **eqn.params)
        if eqn.primitive.multiple_results:
            map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
    map(delete, jaxpr.constvars)
    map(delete, jaxpr.invars)
    map(delete, jaxpr.freevars)
    return map(read, jaxpr.outvars), (env, subenvs)

@lu.transformation_with_aux
def inited_fun(jaxpr, in_tree_def, *args):
    consts, freevar_vals, args = tree_unflatten(in_tree_def, args)
    out = yield (jaxpr, consts, freevar_vals, args), {}
    out = out[0], (out[1],)  # Tuple of envs, not singleton env
    out, out_treedef = tree_flatten(out)
    yield out, out_treedef

def call_init_rule(primitive, params, jaxpr, consts, freevar_vals, in_vals):
    jaxpr, = jaxpr
    consts, = consts
    freevar_vals, = freevar_vals
    all_args, in_treedef = tree_flatten((consts, freevar_vals, in_vals))
    fun = lu.wrap_init(firstpass)
    fun, out_treedef = inited_fun(fun, jaxpr, in_treedef)
    out = primitive.bind(fun, *all_args, **params)
    return tree_unflatten(out_treedef(), out)
init_rules[xla.xla_call_p] = partial(call_init_rule, xla.xla_call_p)

def fastpass(jaxpr, consts, freevar_vals, args, old_env):
    old_env, old_subenvs = old_env

    def read(v):
        if type(v) is jc.Literal:
            return parray(v.val, true_mask(v.val))
        else:
            val = env[repr(v)]
            assert isinstance(val, Parray)
            return val

    def read_old(v):
        if type(v) is jc.Literal:
            return parray(v.val, true_mask(v.val))
        else:
            val = old_env[repr(v)]
            assert isinstance(val, Parray)
            return val

    def read_old_subenvs(vs):
        return old_subenvs[repr(vs)]

    def write(v, val):
        assert isinstance(val, Parray)
        env[repr(v)] = val

    def write_subenvs(vs, env):
        subenvs[repr(vs)] = env

    def delete(v):
        del env[repr(v)]

    env = {}
    subenvs = {}

    write(jc.unitvar, parray(jc.unit, true_mask(jc.unit)))
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    map(write, jaxpr.freevars, freevar_vals)
    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.invars)
        old_outvals = map(read_old, eqn.outvars)
        if eqn.primitive.multiple_results:
            old_ans = old_outvals
        else:
            old_ans = old_outvals[0]
        in_vals = map(read, eqn.invars)
        if eqn.bound_subjaxprs:
            subjaxprs, sub_consts, sub_freevar_vals = unzip3([(
                subjaxpr,
                map(read, const_vars),
                map(read, bound_vars))
                for subjaxpr, const_vars, bound_vars in eqn.bound_subjaxprs])
            old_subenvs_ = read_old_subenvs(tuple(eqn.outvars))
            ans, subenvs_ = get_update(eqn.primitive)(
                eqn.params, subjaxprs, sub_consts, sub_freevar_vals, in_vals,
                old_subenvs_)
            write_subenvs(tuple(eqn.outvars), subenvs_)
        else:
            ans = get_update(eqn.primitive)(old_ans, *in_vals, **eqn.params)
        if eqn.primitive.multiple_results:
            map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
    map(delete, jaxpr.constvars)
    map(delete, jaxpr.invars)
    map(delete, jaxpr.freevars)
    return map(read, jaxpr.outvars), (env, subenvs)

@lu.transformation_with_aux
def updated_fun(jaxpr, in_treedef, *args):
    consts, freevar_vals, args, env = tree_unflatten(in_treedef, args)
    out = yield (jaxpr, consts, freevar_vals, args, env), {}
    out = out[0], (out[1],)  # Tuple of envs, not singleton env
    out, out_treedef = tree_flatten(out)
    yield out, out_treedef

def call_update_rule(
        primitive, params, jaxpr, consts, freevar_vals, in_vals, env):
    jaxpr, = jaxpr
    consts, = consts
    freevar_vals, = freevar_vals
    env, = env
    all_args, in_treedef = tree_flatten((consts, freevar_vals, in_vals, env))
    fun = lu.wrap_init(fastpass)
    fun, out_treedef = updated_fun(fun, jaxpr, in_treedef)
    out = primitive.bind(fun, *all_args, **params)
    return tree_unflatten(out_treedef(), out)
update_rules[xla.xla_call_p] = partial(call_update_rule, xla.xla_call_p)

class Parray(tuple):
    """
    Array which has been partially evaluated, represented by a pair

        (arr, computed)

    where `computed` is a boolean array with True where arr has been computed,
    False where it hasn't yet. `arr` should be initialized to zeros so that
    np.all(~computed & arr == 0).
    """
    pass

# This isn't a pytree node
class ProtectedParray(Parray):
    pass

def _parray_to_iterable(parray):
    if protect_parrays_state:
        return (ProtectedParray(parray),), None
    else:
        arr, mask = parray
        return (arr,), util.HashableMask(mask)
def _iterable_to_parray(mask, arr):
    arr, = arr
    if mask is None:
        return arr
    else:
        return parray(arr, mask.mask)
parray = lambda arr, mask: Parray((arr, mask))
register_pytree_node(Parray, _parray_to_iterable, _iterable_to_parray)

protect_parrays_state = []
@contextmanager
def protect_parrays():
    protect_parrays_state.append(None)
    yield
    protect_parrays_state.pop()

## High level API
def accelerate_part(fun, jit=True):
    def fast_fun(env, *args):
        ans, env = update_env(fun, args, env)
        return ans, Partial(fast_fun, env)
    fast_fun = jit_(fast_fun) if jit else fast_fun

    def first_fun(*args):
        ans, env = init_env(fun, args)
        return ans, Partial(fast_fun, env)
    first_fun = jit_(first_fun) if jit else first_fun
    return first_fun

def accelerate_sections(fixed_point_fun, jit_every=10):
    @jit_
    def accelerated_start(fp_args, x):
        fp = fixed_point_fun(*fp_args)
        x = tree_map(lambda arr: parray(arr, false_mask(arr)), x)
        x, env = init_env(fp, [x])
        i = 1
        while not util.mask_all(x) and i < jit_every:
            x, env = update_env(fp, [x], env)
            i = i + 1
        if util.mask_all(x):
            return x, None
        else:
            return x, Partial(accelerated_section, fp_args, env)

    @jit_
    def accelerated_section(fp_args, env, x):
        fp = fixed_point_fun(*fp_args)
        i = 0
        while not util.mask_all(x) and i < jit_every:
            x, env = update_env(fp, [x], env)
            i = i + 1
        if util.mask_all(x):
            return x, None
        else:
            return x, Partial(accelerated_section, fp_args, env)
    return accelerated_start

@cache()
def fastar_jaxpr(fun, in_tree, in_avals):
    in_pvals = [pe.PartialVal((aval, jc.unit)) for aval in in_avals]
    fun_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    jaxpr, _, consts = pe.trace_to_jaxpr(fun_flat, in_pvals)
    return jaxpr, consts, out_tree()

def init_env(fun, args):
    with protect_parrays():
        args_flat, in_tree = tree_flatten(args)
    assert all(type(arg) is ProtectedParray for arg in args_flat)
    args_flat = [Parray(arg) for arg in args_flat]
    avals = tuple(get_aval(arg) for arg, _ in args_flat)
    jaxpr, consts, out_tree = fastar_jaxpr(fun, in_tree, avals)
    consts = [parray(const, true_mask(const)) for const in consts]
    ans, env = firstpass(jaxpr, consts, [], args_flat)
    return tree_unflatten(out_tree, ans), env

def update_env(fun, args, env):
    with protect_parrays():
        args_flat, in_tree = tree_flatten(args)
    assert all(type(arg) is ProtectedParray for arg in args_flat)
    args_flat = [Parray(arg) for arg in args_flat]
    avals = tuple(get_aval(arg) for arg, _ in args_flat)
    jaxpr, consts, out_tree = fastar_jaxpr(fun, in_tree, avals)
    consts = [parray(const, true_mask(const)) for const in consts]
    ans, env = fastpass(jaxpr, consts, [], args_flat, env)
    return tree_unflatten(out_tree, ans), env
