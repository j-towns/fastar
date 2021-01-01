from dataclasses import dataclass
from typing import Callable, Any

from jax import tree_util
from jax.abstract_arrays import ShapedArray
from jax.core import pytype_aval_mappings
from jax import make_jaxpr
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
  # TODO memoize result
  assert isinstance(delayed, Delayed)
  shape_dtype = delayed.shape_dtype
  if delayed.thunk_tracing:
    return DelayedArray(shape_dtype.shape, shape_dtype.dtype, delayed)
  thunk = lu.wrap_init(delayed.thunk)
  _, in_tree = tree_util.tree_flatten([])
  flat_thunk, out_tree = flatten_fun_nokwargs(thunk, in_tree)
  delayed.thunk_tracing = True
  haxpr = tie_the_knot(fastar_jaxpr(flat_thunk), delayed)
  delayed.thunk_tracing = False
  return out_tree().unflatten(core.eval_haxpr(haxpr.jaxpr, haxpr.consts))
