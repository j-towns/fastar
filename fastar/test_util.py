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
