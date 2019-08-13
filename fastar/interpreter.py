from operator import itemgetter
from collections import OrderedDict

from IPython.display import ProgressBar

from jax.api_util import (pytree_fun_to_jaxtupletree_fun,
                          pytree_to_jaxtupletree, wraps)
import jax.core as jc
from jax.tree_util import build_tree
from jax import tree_util
import jax.interpreters.xla as xla
import jax.interpreters.ad as ad
from jax.interpreters.batching import get_aval
from jax.ad_util import zeros_like_aval
import jax.interpreters.partial_eval as pe
import jax.linear_util as lu
from jax.util import safe_map, safe_zip, unzip2, partial, unzip3
from jax import abstract_arrays
from jax import eval_shape

import fastar.util as util

map = safe_map
zip = safe_zip

class Env(OrderedDict): pass
def env_to_iterable(env):
    keys = tuple(env.keys())
    return tuple(map(env.get, keys)), keys
tree_util.register_pytree_node(Env, env_to_iterable,
                               lambda keys, xs: Env(zip(keys, xs)))

## Core
def make_jaxpr(f):
    def pv_like(x):
        aval = get_aval(x)
        return pe.PartialVal((aval, jc.unit))

    @wraps(f)
    def jaxpr_maker(*args, **kwargs):
        fun = lu.wrap_init(f)
        jax_args, in_trees = unzip2(map(pytree_to_jaxtupletree, args))
        jaxtree_fun, out_tree = pytree_fun_to_jaxtupletree_fun(fun, in_trees)
        pvals = map(pv_like, jax_args)
        jaxpr, _, consts = pe.trace_to_jaxpr(jaxtree_fun, pvals, **kwargs)
        return jaxpr, consts

    jaxpr_maker.__name__ = "make_jaxpr({})".format(jaxpr_maker.__name__)
    return jaxpr_maker

init_rules = {}
def init_ans(prim, *args, **params):
    if prim in init_rules:
        return init_rules[prim](*args, **params)
    else:
        args = map(lambda arg: get_aval(arg[0]), args)
        abstract_out = prim.abstract_eval(*args, **params)
        return parray(zeros_like_aval(abstract_out),
                       util.false_mask(abstract_out))

update_rules = {}
def get_update(p):
    try:
        return update_rules[p]
    except KeyError:
        raise NotImplementedError(
            "Parray computation rule for '{}' not implemented".format(p))

# Populate the cache
def firstpass(jaxpr, consts, freevar_vals, args):
    def read(v):
        if type(v) is jc.Literal:
            return parray(v.val, util.true_mask(v.val))
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
    subenvs = {}

    write(jc.unitvar, parray(jc.unit, util.true_mask(jc.unit)))
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    map(write, jaxpr.freevars, freevar_vals)
    for eqn in jaxpr.eqns:
        if not eqn.restructure:
            in_vals = map(read, eqn.invars)
        else:
            in_vals = [pack(map(read, invars)) if type(invars) is tuple
                       else read(invars) for invars in eqn.invars]
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
                eqn.params, subjaxprs, sub_consts,
                sub_freevar_vals, in_vals, subenvs_)

            write_subenvs(tuple(eqn.outvars), subenvs_)
        else:
            ans = init_ans(eqn.primitive, *in_vals, **eqn.params)
            ans = get_update(eqn.primitive)(ans, *in_vals, **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar), (env, subenvs)

def wrap(arr, mask):
    return parray(arr, mask.mask)

@lu.transformation_with_aux
def inited_fun(jaxpr, in_tree_def, masks, mask_tree_def, args):
    consts, freevar_vals, args = args
    consts, freevar_vals, args = tree_util.build_tree(
        in_tree_def, (consts, freevar_vals, args))
    const_masks, freevar_masks, arg_masks = tree_util.build_tree(mask_tree_def, masks)
    freevar_vals = tree_util.tree_multimap(wrap, freevar_vals, freevar_masks)
    args = tree_util.tree_multimap(wrap, args, arg_masks)
    consts = tree_util.tree_multimap(wrap, consts, const_masks)
    out = yield (jaxpr, consts, freevar_vals, args), {}
    out = out[0], (out[1],)  # Tuple of envs, not singleton env
    out, out_mask = util.tree_unmask(out)
    out_jtuple, tree_def = ad.tree_to_jaxtuples(out)
    yield out_jtuple, (out_mask, tree_def)

def call_init_rule(primitive, params, jaxpr, consts, freevar_vals, in_vals):
    jaxpr, = jaxpr
    consts, = consts
    freevar_vals, = freevar_vals
    (consts, freevar_vals, in_vals), masks, mask_tree_def = (
        util.tree_unmask_hashably((consts, freevar_vals, in_vals)))
    (consts, freevar_vals, in_vals), in_tree_def = ad.tree_to_jaxtuples(
        (consts, freevar_vals, in_vals))
    fun = lu.wrap_init(firstpass)
    fun, out_mask_and_tree_def = inited_fun(fun, jaxpr, in_tree_def,
                                            masks, mask_tree_def)
    all_args = jc.pack((consts, freevar_vals, in_vals))
    out = primitive.bind(fun, all_args, **params)
    out_mask, out_tree_def = out_mask_and_tree_def()
    out = tree_util.build_tree(out_tree_def, out)
    return tree_util.tree_multimap(lambda arr, mask: parray(arr, mask),
                                   out, out_mask)

init_rules[xla.xla_call_p] = partial(call_init_rule, xla.xla_call_p)

def fastpass(jaxpr, consts, freevar_vals, args, old_env):
    old_env, old_subenvs = old_env

    def read(v):
        if type(v) is jc.Literal:
            return parray(v.val, util.true_mask(v.val))
        else:
            val = env[v]
            assert isinstance(val, Parray)
            return val

    def read_old(v):
        if type(v) is jc.Literal:
            return parray(v.val, util.true_mask(v.val))
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

    env = Env()
    subenvs = {}

    write(jc.unitvar, parray(jc.unit, util.true_mask(jc.unit)))
    map(write, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    map(write, jaxpr.freevars, freevar_vals)
    for eqn in jaxpr.eqns:
        if not eqn.restructure:
            in_vals = map(read, eqn.invars)
        else:
            in_vals = [pack(map(read, invars)) if type(invars) is tuple
                       else read(invars) for invars in eqn.invars]
        old_outvals = map(read_old, eqn.outvars)
        old_ans = old_outvals if eqn.destructure else old_outvals[0]
        in_vals = map(read, eqn.invars)
        if eqn.bound_subjaxprs:
            subjaxprs, sub_consts, sub_freevar_vals = unzip3([(
                subjaxpr,
                map(read, const_vars),
                map(read, bound_vars))
                for subjaxpr, const_vars, bound_vars
                in eqn.bound_subjaxprs])
            old_subenvs_ = read_old_subenvs(tuple(eqn.outvars))
            ans, subenvs_ = get_update(eqn.primitive)(
                eqn.params, subjaxprs, sub_consts,
                sub_freevar_vals, in_vals, old_subenvs_)
            write_subenvs(tuple(eqn.outvars), subenvs_)
        else:
            ans = get_update(eqn.primitive)(old_ans, *in_vals, **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar), (env, subenvs)

@lu.transformation_with_aux
def updated_fun(jaxpr, in_tree_def, masks, mask_tree_def, args):
    consts, freevar_vals, args, env = args
    consts, freevar_vals, args, env = tree_util.build_tree(
        in_tree_def, (consts, freevar_vals, args, env))
    const_masks, freevar_masks, arg_masks, env_masks = tree_util.build_tree(
        mask_tree_def, masks)
    consts = tree_util.tree_multimap(wrap, consts, const_masks)
    freevar_vals = tree_util.tree_multimap(wrap, freevar_vals, freevar_masks)
    args = tree_util.tree_multimap(wrap, args, arg_masks)
    env = tree_util.tree_multimap(wrap, env, env_masks)
    out = yield (jaxpr, consts, freevar_vals, args, env), {}
    out = out[0], (out[1],)  # Tuple of envs, not singleton env
    out, out_mask = util.tree_unmask(out)
    out_jtuple, tree_def = ad.tree_to_jaxtuples(out)
    yield out_jtuple, (out_mask, tree_def)

def call_update_rule(primitive, params, jaxpr, consts, freevar_vals, in_vals,
                     env):
    jaxpr, = jaxpr
    consts, = consts
    freevar_vals, = freevar_vals
    env, = env
    (consts, freevar_vals, in_vals, env), masks, mask_tree_def =(
        util.tree_unmask_hashably((consts, freevar_vals, in_vals, env)))
    (consts, freevar_vals, in_vals, env), in_tree_def = ad.tree_to_jaxtuples(
        (consts, freevar_vals, in_vals, env))
    fun = lu.wrap_init(fastpass)
    fun, out_mask_and_tree_def = updated_fun(
        fun, jaxpr, in_tree_def, masks, mask_tree_def)
    all_args = jc.pack((consts, freevar_vals, in_vals, env))
    out = primitive.bind(fun, all_args, **params)
    out_mask, out_tree_def = out_mask_and_tree_def()
    out = tree_util.build_tree(out_tree_def, out)
    return tree_util.tree_multimap(lambda arr, mask: parray(arr, mask),
                                   out, out_mask)
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
parray = lambda arr, mask: parray(arr, mask)

## High level API
def accelerate(fun):
    def first_fun(*args, **kwargs):
        raw_args = map(lambda arg: arg[0], args)
        jaxpr, consts = make_jaxpr(fun)(*raw_args, **kwargs)
        consts = tree_util.tree_multimap(parray, consts, util.true_mask(consts))
        args_to_numpy = lambda args: [parray(util.to_numpy(raw_arg), mask)
                                      for raw_arg, mask in args]

        ans, env = firstpass(jaxpr, consts, [], args_to_numpy(args))
        def fast_fun(env, *args, **kwargs):
            ans, env = fastpass(jaxpr, consts, [], args_to_numpy(args), env)
            return ans, partial(fast_fun, env)
        return ans, partial(fast_fun, env)
    return first_fun

def accelerate_fixed_point(fp, max_iter=1000):
    def accelerated(x):
        p = ProgressBar(x[1].size)
        jaxpr, consts = make_jaxpr(fp)(x)
        x = tree_util.tree_multimap(parray, x, util.false_mask(x))
        consts = tree_util.tree_multimap(parray, consts, util.true_mask(consts))
        x, env = firstpass(jaxpr, consts, [], (x,))
        i = 0
        while not util.mask_all(x) and i < max_iter:
            p.progress = np.sum(x[1])
            x, env = fastpass(jaxpr, consts, (x,), env)
            i = i + 1
        x, _ = x
        return x
    return accelerated
