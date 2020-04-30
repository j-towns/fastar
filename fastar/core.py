from contextlib import contextmanager
from functools import reduce
from operator import and_

import numpy as onp

import jax.core as jc
from jax import abstract_arrays
from jax.abstract_arrays import ShapedArray
from jax import dtypes
import jax.interpreters.partial_eval as pe
import jax.lax as lax
import jax.linear_util as lu
import jax.scipy.special as special
import numpy as onp
from jax import numpy as np, jit as jit_
from jax.ad_util import zeros_like_aval
from jax.api_util import flatten_fun_nokwargs
from .util import true_mask, false_mask, mask_all, Hashable, mask_to_slices
from jax.interpreters import xla
from jax.ops import index_update
from jax.tree_util import (
  tree_flatten, tree_unflatten, tree_map, register_pytree_node)
from jax.util import curry, safe_zip, safe_map
from jax.util import partial, unzip2, cache

map = safe_map
zip = safe_zip


def _get_aval(x):
  return abstract_arrays.raise_to_shaped(jc.get_aval(x))


## Core
init_rules = {}


def _init_ans(prim, *args, **params):
  if prim in init_rules:
    return init_rules[prim](*args, **params)
  else:
    args = [_get_aval(arg) for arg, _ in args]
    abstract_out = prim.abstract_eval(*args, **params)
    if prim.multiple_results:
      return [parray(zeros_like_aval(o), false_mask(o))
              for o in abstract_out]
    else:
      return parray(zeros_like_aval(abstract_out),
                    false_mask(abstract_out))


update_rules = {}


def _get_update(p):
  try:
    return update_rules[p]
  except KeyError:
    raise NotImplementedError(
      "Fastar update rule for '{}' not implemented".format(p))


# Populate the cache
def _firstpass(jaxpr, consts, args):
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

  def write_subenv(vs, env):
    subenvs[repr(vs)] = env

  # Mapping from repr(var) to var's value. We use repr(var) because it is
  # hashable, and we need env to be a valid pytree.
  env = {}
  subenvs = {}

  write(jc.unitvar, parray(jc.unit, true_mask(jc.unit)))
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      ans, subenv = _init_ans(eqn.primitive, params, call_jaxpr, in_vals)
      ans, subenv = _get_update(eqn.primitive)(
        params, call_jaxpr, in_vals, subenv)

      write_subenv(tuple(eqn.outvars), subenv)
    else:
      ans = _init_ans(eqn.primitive, *in_vals, **eqn.params)
      ans = _get_update(eqn.primitive)(ans, *in_vals, **eqn.params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  map(delete, jaxpr.constvars)
  map(delete, jaxpr.invars)
  return map(read, jaxpr.outvars), (env, subenvs)


@lu.transformation_with_aux
def _inited_fun(jaxpr, in_tree_def, knowns, *args):
  knowns = [known.val for known in knowns]
  args = tree_unflatten(in_tree_def, map(parray, args, knowns))
  out = yield (jaxpr, [], args), {}
  out, out_treedef = tree_flatten(out)
  out, out_known = unzip2(out)
  yield out, (out_treedef, out_known)


def _call_init_rule(primitive, params, jaxpr, in_vals):
  args, in_treedef = tree_flatten(in_vals)
  args, knowns = unzip2(args)
  knowns = tuple(map(Hashable, knowns))
  fun = lu.wrap_init(_firstpass)
  fun, aux = _inited_fun(fun, jaxpr, in_treedef, knowns)
  out = primitive.bind(fun, *args, **params)
  out_treedef, out_known = aux()
  return tree_unflatten(out_treedef, map(parray, out, out_known))


init_rules[xla.xla_call_p] = partial(_call_init_rule, xla.xla_call_p)


def _fastpass(jaxpr, consts, args, old_env):
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

  def read_old_subenv(vs):
    return old_subenvs[repr(vs)]

  def write(v, val):
    assert isinstance(val, Parray)
    env[repr(v)] = val

  def write_subenv(vs, env):
    subenvs[repr(vs)] = env

  def delete(v):
    del env[repr(v)]

  env = {}
  subenvs = {}

  write(jc.unitvar, parray(jc.unit, true_mask(jc.unit)))
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_vals = map(read, eqn.invars)
    call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
    old_outvals = map(read_old, eqn.outvars)
    if eqn.primitive.multiple_results:
      old_ans = old_outvals
    else:
      old_ans = old_outvals[0]
    in_vals = map(read, eqn.invars)
    if call_jaxpr:
      old_subenv = read_old_subenv(tuple(eqn.outvars))
      ans, subenv = _get_update(eqn.primitive)(
        params, call_jaxpr, in_vals, old_subenv)
      write_subenv(tuple(eqn.outvars), subenv)
    else:
      ans = _get_update(eqn.primitive)(old_ans, *in_vals, **eqn.params)
    if eqn.primitive.multiple_results:
      map(write, eqn.outvars, ans)
    else:
      write(eqn.outvars[0], ans)
  map(delete, jaxpr.constvars)
  map(delete, jaxpr.invars)
  return map(read, jaxpr.outvars), (env, subenvs)


@lu.transformation_with_aux
def _updated_fun(jaxpr, in_treedef, knowns, *all_args):
  knowns = [known.val for known in knowns]
  args, env = tree_unflatten(in_treedef, map(parray, all_args, knowns))
  out = yield (jaxpr, [], args, env), {}
  out, out_treedef = tree_flatten(out)
  out, out_known = unzip2(out)
  yield out, (out_treedef, out_known)


def _call_update_rule(primitive, params, jaxpr, in_vals, env):
  all_args, in_treedef = tree_flatten((in_vals, env))
  all_args, knowns = unzip2(all_args)
  knowns = tuple(map(Hashable, knowns))
  fun = lu.wrap_init(_fastpass)
  fun, aux = _updated_fun(fun, jaxpr, in_treedef, knowns)
  out = primitive.bind(fun, *all_args, **params)
  out_treedef, out_known = aux()
  return tree_unflatten(out_treedef, map(parray, out, out_known))


update_rules[xla.xla_call_p] = partial(_call_update_rule, xla.xla_call_p)


class Parray(tuple):
  """
  Array which has been partially evaluated, represented by a pair
      (arr, known)
  where `known` is a boolean array with True where arr has been computed, and
  values are fixed.  False where it hasn't yet. `arr` should be initialized to
  zeros so that np.all(~known & arr == 0).
  """
  # TODO(j-towns): do we need to require uncomputed values to be zero?
  pass
parray = lambda arr, mask: Parray((arr, mask))


@cache()
def _fastar_jaxpr(fun, in_tree, in_avals):
  in_pvals = [pe.PartialVal((aval, jc.unit)) for aval in in_avals]
  fun_flat, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  jaxpr, _, consts = pe.trace_to_jaxpr(fun_flat, in_pvals, stage_out=True)
  return jaxpr, consts, out_tree()


def _init_env(fun, args):
  args_flat, in_tree = tree_flatten(args)
  assert all(type(arg) is Parray for arg in args_flat)
  avals = tuple(_get_aval(arg) for arg, _ in args_flat)
  jaxpr, consts, out_tree = _fastar_jaxpr(fun, in_tree, avals)
  consts = [parray(const, true_mask(const)) for const in consts]
  ans, env = _firstpass(jaxpr, consts, args_flat)
  # TODO(j-towns) could type-check ans
  return tree_unflatten(out_tree, ans), env


def _update_env(fun, args, env):
  args_flat, in_tree = tree_flatten(args)
  assert all(type(arg) is Parray for arg in args_flat)
  avals = tuple(_get_aval(arg) for arg, _ in args_flat)
  jaxpr, consts, out_tree = _fastar_jaxpr(fun, in_tree, avals)
  # TODO(j-towns): Shouldn't need to re-wrap consts here
  consts = [parray(const, true_mask(const)) for const in consts]
  ans, env = _fastpass(jaxpr, consts, args_flat, env)
  # TODO(j-towns) could type-check ans
  return tree_unflatten(out_tree, ans), env


## High level API
def accelerate(fun):
  """Accelerate the execution of `fun` when `fun` is called many times.

  The input function is assumed to be part of a loop where intermediate values
  are progressively evaluated, for example an autoregressive sampling loop. The
  loop can be accelerated using elementwise partial evaluation. Before each call
  to fun, each element of each array in *args is either 'known' or 'unknown'.
  This state is indicated using a boolean mask, paired with each array. The pair
  (arr, mask) must be wrapped in a `Parray`.

  We require that each time fun is called, the inputs are at least as known as
  the previous call, and that once an element is known its value cannot change.
  Specifically if (arr_new, mask_new) are the current inputs and (arr_old,
  mask_old) are the inputs to the previous call, we require that

    1) mask_new >= mask_old                     (monotonically increasing mask)

  and

    2) all(arr_new[mask_old] == arr_old[mask_old])                (consistency)

  Thus all intermediates can progressively become known (by updating elements of
  a cache) and the outputs of the function also progressively become known. Each
  element of the cache, and the outputs, are returned in Parray format (i.e. as
  a pytree containing Parray(arr, mask) pairs.

  Args:
    fun: Function to be accelerated.

  Returns:
    A pair (init, update), both functions, init taking in *args which are the
      same as the arguments for fun, but replacing each array with a Parray,
      that is a pair (arr, mask), where mask indicates which values are 'known'.
      init returns a pair (cache, out). The update takes arguments cache, *args,
      with cache in the format returned by init, and args in the same format as
      described for init. The update returns an updated (cache, out) pair.
  """
  def init(*args):
    """
    Initializes and performs first update, returning (output, cache).
    """
    return _init_env(fun, args)

  def update(cache, *args):
    """Returns updated (output, cache)."""
    return _update_env(fun, args, cache)

  return init, update
