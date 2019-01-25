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
            val = Masked((val, util.false_mask(val)))
        return val

    def write_const(v, val):
        val = Masked((val, util.true_mask(val)))
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


## Subcomputation rules
sub_rules = {}

def get_subprimitive(p):
    try:
        return sub_rules[p]
    except KeyError:
        raise NotImplementedError(
            "Masked computation rule for '{}' not implemented".format(p))
