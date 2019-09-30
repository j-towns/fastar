import time
from operator import itemgetter
from collections import OrderedDict

from IPython.display import ProgressBar

from jax.api_util import flatten_fun, wraps, flatten_fun_nokwargs
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import jax.core as jc
from jax.tree_util import build_tree
from jax import tree_util
import jax.interpreters.xla as xla
import jax.interpreters.ad as ad
from jax.ad_util import zeros_like_aval
import jax.interpreters.partial_eval as pe
import jax.linear_util as lu
from jax.util import safe_map, safe_zip, unzip2, partial, unzip3, cache
from jax import abstract_arrays
from jax import eval_shape

import fastar.util as util
from fastar.util import true_mask, false_mask

map = safe_map
zip = safe_zip

def get_aval(x):
    return abstract_arrays.raise_to_shaped(jc.get_aval(x))

class Env(OrderedDict): pass
def env_to_iterable(env):
    keys = tuple(env.keys())
    return tuple(map(env.get, keys)), keys
tree_util.register_pytree_node(Env, env_to_iterable,
                               lambda keys, xs: Env(zip(keys, xs)))

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
            val = env[v]
            assert isinstance(val, Parray)
            return val

    def write(v, val):
        assert isinstance(val, Parray)
        env[v] = val

    def write_subenvs(vs, env):
        subenvs[vs] = env

    env = Env()
    subenvs = Env()

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
                eqn.params, subjaxprs, sub_freevar_vals, in_vals, subenvs_)

            write_subenvs(tuple(eqn.outvars), subenvs_)
        else:
            ans = init_ans(eqn.primitive, *in_vals, **eqn.params)
            ans = get_update(eqn.primitive)(ans, *in_vals, **eqn.params)
        if eqn.primitive.multiple_results:
            map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
    return map(read, jaxpr.outvars), (env, subenvs)

def wrap(arr, mask):
    return parray(arr, mask.mask)

@lu.transformation_with_aux
def inited_fun(jaxpr, in_tree_def, masks, *args):
    args = [parray(a, m.mask) for a, m in zip(args, masks)]
    consts, freevar_vals, args = tree_unflatten(in_tree_def, args)
    out = yield (jaxpr, consts, freevar_vals, args), {}
    out = out[0], (out[1],)  # Tuple of envs, not singleton env
    out, out_mask, out_treedef = util.unmask_and_flatten(out)
    yield out, (out_mask, out_treedef)

def call_init_rule(primitive, params, jaxpr, consts, freevar_vals, in_vals):
    jaxpr, = jaxpr
    consts, = consts
    freevar_vals, = freevar_vals
    all_args, masks, in_treedef = util.unmask_and_flatten(
        (consts, freevar_vals, in_vals))
    masks = tuple(map(util.HashableMask, masks))
    fun = lu.wrap_init(firstpass)
    fun, out_mask_and_treedef = inited_fun(fun, jaxpr, in_treedef, masks)
    out = primitive.bind(fun, *all_args, **params)
    out_mask, out_treedef = out_mask_and_treedef()
    return tree_unflatten(out_treedef, map(parray, out, out_mask))
init_rules[xla.xla_call_p] = partial(call_init_rule, xla.xla_call_p)

def fastpass(jaxpr, freevar_vals, args, old_env):
    old_env, old_subenvs = old_env

    def read(v):
        if type(v) is jc.Literal:
            return parray(v.val, true_mask(v.val))
        else:
            val = env[v]
            assert isinstance(val, Parray)
            return val

    def read_old(v):
        if type(v) is jc.Literal:
            return parray(v.val, true_mask(v.val))
        else:
            val = old_env[v]
            assert isinstance(val, Parray)
            return val

    def read_old_subenvs(vs):
        return old_subenvs[vs]

    def write(v, val):
        assert isinstance(val, Parray)
        env[v] = val

    def write_subenvs(vs, env):
        subenvs[vs] = env

    def copy_old_to_new(v):
        write(v, read_old(v))

    env = Env()
    subenvs = Env()

    write(jc.unitvar, parray(jc.unit, true_mask(jc.unit)))
    map(copy_old_to_new, jaxpr.constvars)
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
            subjaxprs, sub_freevar_vals = unzip2([(
                subjaxpr,
                map(read, bound_vars))
                for subjaxpr, _, bound_vars in eqn.bound_subjaxprs])
            old_subenvs_ = read_old_subenvs(tuple(eqn.outvars))
            ans, subenvs_ = get_update(eqn.primitive)(
                eqn.params, subjaxprs, sub_freevar_vals, in_vals, old_subenvs_)
            write_subenvs(tuple(eqn.outvars), subenvs_)
        else:
            ans = get_update(eqn.primitive)(old_ans, *in_vals, **eqn.params)
        if eqn.primitive.multiple_results:
            map(write, eqn.outvars, ans)
        else:
            write(eqn.outvars[0], ans)
    return map(read, jaxpr.outvars), (env, subenvs)

@lu.transformation_with_aux
def updated_fun(jaxpr, in_treedef, masks, *args):
    args = [parray(a, m.mask) for a, m in zip(args, masks)]
    freevar_vals, args, env = tree_unflatten(in_treedef, args)
    out = yield (jaxpr, freevar_vals, args, env), {}
    out = out[0], (out[1],)  # Tuple of envs, not singleton env
    out, out_mask, out_treedef = util.unmask_and_flatten(out)
    yield out, (out_mask, out_treedef)

def call_update_rule(primitive, params, jaxpr, freevar_vals, in_vals, env):
    jaxpr, = jaxpr
    freevar_vals, = freevar_vals
    env, = env
    all_args, masks, in_treedef = util.unmask_and_flatten(
        (freevar_vals, in_vals, env))
    masks = tuple(map(util.HashableMask, masks))
    fun = lu.wrap_init(fastpass)
    fun, out_mask_and_treedef = updated_fun(fun, jaxpr, in_treedef, masks)
    out = primitive.bind(fun, *all_args, **params)
    out_mask, out_treedef = out_mask_and_treedef()
    return tree_unflatten(out_treedef, map(parray, out, out_mask))
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
parray = lambda arr, mask: Parray((arr, mask))

## High level API
def accelerate_part(fun):
    def first_fun(*args):
        ans, env = init_env(fun, args)
        def fast_fun(env, *args):
            ans, env = update_env(fun, args, env)
            return ans, partial(fast_fun, env)
        return ans, partial(fast_fun, env)
    return first_fun

@cache()
def fastar_jaxpr(fun, in_tree, in_avals):
    in_pvals = [pe.PartialVal((aval, jc.unit)) for aval in in_avals]
    fun_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
    jaxpr, _, consts = pe.trace_to_jaxpr(fun_flat, in_pvals)
    return jaxpr, consts, out_tree()

def init_env(fun, args):
    args_flat, in_tree = tree_flatten(args)
    avals = tuple(get_aval(arg) for arg, _ in args_flat)
    jaxpr, consts, out_tree = fastar_jaxpr(fun, in_tree, avals)
    consts = [parray(const, true_mask(const)) for const in consts]
    ans, env = firstpass(jaxpr, consts, [], args_flat)
    return tree_unflatten(out_tree, ans), env

def update_env(fun, args, env):
    args_flat, in_tree = tree_flatten(args)
    avals = tuple(get_aval(arg) for arg, _ in args_flat)
    jaxpr, _, out_tree = fastar_jaxpr(fun, in_tree, avals)
    ans, env = fastpass(jaxpr, [], args_flat, env)
    return tree_unflatten(out_tree, ans), env

def accelerate(fixed_point_fun, max_iter=1000):
    def accelerated(x):
        x = tree_map(lambda x_leaf: parray(x_leaf, false_mask(x_leaf)), x)
        x, env = init_env(fixed_point_fun, [x])
        i = 1
        while not util.tree_mask_all(x) and i < max_iter:
            x, env = update_env(fixed_point_fun, [x], env)
            i = i + 1
        return tree_map(lambda x_leaf: x_leaf[0], x)
    return accelerated
