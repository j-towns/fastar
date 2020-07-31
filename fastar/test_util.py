from itertools import chain
from random import shuffle
import numpy as np
from jax import numpy as jnp, test_util as jtu
from jax.util import safe_map, safe_zip
from jax.tree_util import tree_multimap, tree_flatten, tree_map

from fastar import lazy_eval, lazy_eval_fixed_point, LazyArray

map = safe_map
zip = safe_zip


def check_shape_and_dtype(expected, actual):
  assert expected.shape == actual.shape
  assert expected.dtype == actual.dtype

def naive_fixed_point(fun, arg):
  arg, arg_prev = fun(arg), arg
  while not jnp.all(arg == arg_prev):
    arg, arg_prev = fun(arg), arg
  return arg

def check_child_counts(arrs):
  visited = set()
  def _check_child_counts(arrs):
    for arr in arrs:
      if isinstance(arr, LazyArray) and arr not in visited:
        assert type(arr.child_counts) is np.ndarray
        assert arr.child_counts.dtype == np.int64
        assert np.all(arr.child_counts == 0)
        visited.add(arr)
        _check_child_counts(arr.eqn.invars)
  _check_child_counts(arrs)

def check_state(arrs):
  # Make sure none of the elements are in the temporary REQUESTED state
  visited = set()
  def _check_state(arrs):
    for arr in arrs:
      if isinstance(arr, LazyArray) and arr not in visited:
        assert np.all((arr.state == 0) | (arr.state == 1))
        visited.add(arr)
        _check_state(arr.eqn.invars)
  _check_state(arrs)

def _identity(x):
  return x + np.zeros((), x.dtype)

def check_lazy_fun(fun_, *args, atol=None, rtol=None):
  def fun(*args):
    args = tree_map(_identity, args)
    return fun_(*args)

  out_expected_flat, out_expected_tree = tree_flatten(fun(*args))
  out_flat, out_tree = tree_flatten(lazy_eval(fun, *args))
  assert out_expected_tree == out_tree
  tree_multimap(check_shape_and_dtype, out_expected_flat, out_flat)
  jtu.check_close(out_expected_flat,
                  [o[:] if o.shape else o[()] for o in out_flat], atol, rtol)
  check_child_counts(out_flat)
  check_state(out_flat)
  out_flat, _ = tree_flatten(lazy_eval(fun, *args))
  indices = []
  for n, o in enumerate(out_flat):
    indices.append([(n, i) for i in np.ndindex(*o.shape)])
  indices = list(chain(*indices))
  shuffle(indices)
  indices = indices[:5]
  for n, i in indices:
    jtu.check_close(out_flat[n][i], out_expected_flat[n][i], atol, rtol)
    assert np.dtype(out_flat[n][i]) == np.dtype(out_expected_flat[n][i])
  check_child_counts(out_flat)
  check_state(out_flat)

def check_lazy_fixed_point(fun, mock_arg, atol=None, rtol=None):
  out_expected_flat, out_expected_tree = tree_flatten(
      naive_fixed_point(fun, mock_arg))
  out_flat, out_tree = tree_flatten(lazy_eval_fixed_point(fun, mock_arg))
  assert out_expected_tree == out_tree
  tree_multimap(check_shape_and_dtype, out_expected_flat, out_flat)
  jtu.check_close(out_expected_flat, [o[:] for o in out_flat], atol, rtol)
  check_child_counts(out_flat)
  check_state(out_flat)
  out_flat, out_tree = tree_flatten(lazy_eval_fixed_point(fun, mock_arg))
  indices = []
  for n, o in enumerate(out_flat):
    indices.append([(n, i) for i in np.ndindex(*o.shape)])
  indices = list(chain(*indices))
  shuffle(indices)
  indices = indices[:5]
  for n, i in indices:
    jtu.check_close(out_flat[n][i], out_expected_flat[n][i], atol, rtol)
    assert np.dtype(out_flat[n][i]) == np.dtype(out_expected_flat[n][i])
  check_child_counts(out_flat)
  check_state(out_flat)
