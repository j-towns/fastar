from operator import itemgetter

from jax.api_util import (pytree_fun_to_jaxtupletree_fun,
                          pytree_to_jaxtupletree, wraps)
import jax.core as jc
import jax.interpreters.xla as xla
import jax.interpreters.partial_eval as pe
import jax.linear_util as lu
from jax.util import safe_map, unzip2, partial

import fastar.util as util

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
    def read(v):
        val = env[v]
        assert isinstance(val, Parray)
        return val

    def write(v, val):
        assert isinstance(val, Parray)
        env[v] = val

    def write_const(v, val):
        val = Parray((val, util.true_mask(val)))
        env[v] = val

    env = {}
    write_const(jc.unitvar, jc.unit)
    map(write_const, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        in_vals = map(read, eqn.invars)
        if eqn.bound_subjaxprs:
            raise NotImplementedError
        ans = Parray(util.init_ans(eqn.primitive, *in_vals, **eqn.params))
        ans = get_update(eqn.primitive)(ans, *in_vals, **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar), env


def fastpass(jaxpr, consts, old_env, *args):
    def read(v):
        val = env[v]
        assert isinstance(val, Parray)
        return val

    def read_old(v):
        val = old_env[v]
        assert isinstance(val, Parray)
        return val

    def write(v, val):
        assert isinstance(val, Parray)
        env[v] = val

    def write_const(v, val):
        val = Parray((val, util.true_mask(val)))
        env[v] = val

    env = {}
    write_const(jc.unitvar, jc.unit)
    map(write_const, jaxpr.constvars, consts)
    map(write, jaxpr.invars, args)
    for eqn in jaxpr.eqns:
        old_outvals = map(read_old, eqn.outvars)
        old_ans = old_outvals if eqn.destructure else old_outvals[0]
        in_vals = map(read, eqn.invars)
        if eqn.bound_subjaxprs:
            raise NotImplementedError
        ans = get_update(eqn.primitive)(old_ans, *in_vals, **eqn.params)
        outvals = list(ans) if eqn.destructure else [ans]
        map(write, eqn.outvars, outvals)
    return read(jaxpr.outvar), env

class Parray(tuple):
    """
    Array which has been partially evaluated, represented by a pair

        (arr, computed)

    where `computed` is a boolean array with True where arr has been computed,
    False where it hasn't yet. `arr` should be initialized to zeros so that
    np.all(~computed & arr == 0).
    """
    pass

## Update rules
update_rules = {}

def get_update(p):
    try:
        return update_rules[p]
    except KeyError:
        raise NotImplementedError(
            "Parray computation rule for '{}' not implemented".format(p))

## High level API
def accelerate(fun):
    def first_fun(*args, **kwargs):
        raw_args = map(lambda arg: arg[0], args)
        jaxpr, consts = make_jaxpr(fun)(*raw_args, **kwargs)
        args_to_numpy = lambda args: [Parray((util.to_numpy(raw_arg), mask))
                                      for raw_arg, mask in args]

        ans, env = firstpass(jaxpr, consts, *args_to_numpy(args))

        def fast_fun(env, *args, **kwargs):
            ans, env = fastpass(jaxpr, consts, env, *args_to_numpy(args))
            return ans, partial(fast_fun, env)
        return ans, partial(fast_fun, env)
    return first_fun
