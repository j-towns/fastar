from dataclasses import dataclass
from typing import Callable, Any

from jax.abstract_arrays import ShapedArray
from jax.core import pytype_aval_mappings
from jax import make_jaxpr, tree_flatten
import jax.linear_util as lu
from jax.api_util import flatten_fun_nokwargs

from fastar import core
from fastar.jaxpr_util import fastar_jaxpr, tie_the_knot, DelayedArray


def lazy_eval(fun, *args):
  args_flat, in_tree = tree_util.tree_flatten(args)
  f = lu.wrap_init(fun)
  flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
  jaxpr, consts, _, out_avals = fastar_jaxpr(flat_fun, *args_flat)
  outs_flat = core.lazy_eval_jaxpr(jaxpr, consts, *args_flat)
  for out, aval in zip(outs_flat, out_avals):
    assert core.get_aval(out) == aval
  return tree_util.tree_unflatten(out_tree(), outs_flat)

def lazy_eval_fixed_point(fun, mock_arg):
  arg_flat, in_tree = tree_util.tree_flatten([mock_arg])
  f = lu.wrap_init(fun)
  flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
  jaxpr, consts, _, out_avals = tie_the_knot(
      fastar_jaxpr(flat_fun, *arg_flat))
  outs_flat = core.lazy_eval_jaxpr(jaxpr, consts)
  for out, aval in zip(outs_flat, out_avals):
    assert core.get_aval(out) == aval
  return tree_util.tree_unflatten(out_tree(), outs_flat)

@dataclass
class Delayed:
  thunk: Callable
  shape_dtype: Any
  thunk_tracing: bool = False

def delay(thunk, result_shape_dtype):
  return Delayed(thunk, result_shape_dtype)

def force(delayed):
  # TODO maybe memoize result
  assert isinstance(delayed, Delayed)
  shape_dtypes, out_tree_ = tree_flatten(delayed.shape_dtype)
  if delayed.thunk_tracing:
    return out_tree_.unflatten([
        DelayedArray(sd.shape, sd.dtype, delayed, i)
        for i, sd in enumerate(shape_dtypes)])
  thunk = lu.wrap_init(delayed.thunk)
  _, in_tree = tree_flatten([])
  flat_thunk, out_tree = flatten_fun_nokwargs(thunk, in_tree)
  delayed.thunk_tracing = True
  jaxpr = fastar_jaxpr(flat_thunk)
  assert all((a.shape, a.dtype) == (b.shape, b.dtype)
             for a, b in zip(shape_dtypes, jaxpr.out_avals))
  haxpr = tie_the_knot(jaxpr, delayed)
  delayed.thunk_tracing = False
  assert out_tree() == out_tree_
  return out_tree().unflatten(core.eval_haxpr(haxpr.jaxpr, haxpr.consts))
