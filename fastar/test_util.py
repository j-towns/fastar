from functools import partial
from itertools import chain
from random import shuffle
import jax.lax as lax
import numpy as np
import numpy.testing as np_testing
import jax.numpy as jnp
import jax.test_util as jtu
from jax import vjp
from jax.util import safe_map, safe_zip
from jax.tree_util import tree_multimap, tree_flatten

from fastar import lazy_eval

map = safe_map
zip = safe_zip


def check_shape_and_dtype(expected, actual):
  assert expected.shape == actual.shape
  assert expected.dtype == actual.dtype

def check_lazy_fun(fun, *args):
  out_expected_flat, out_expected_tree = tree_flatten(fun(*args))
  out_flat, out_tree = tree_flatten(lazy_eval(fun, *args))
  assert out_expected_tree == out_tree
  tree_multimap(check_shape_and_dtype, out_expected_flat, out_flat)
  jtu.check_eq(out_expected_flat, [o[:] for o in out_flat])
  out_flat, _ = tree_flatten(lazy_eval(fun, *args))
  indices = []
  for n, o in enumerate(out_flat):
    indices.append([(n, i) for i in np.ndindex(*o.shape)])
  indices = list(chain(*indices))
  shuffle(indices)
  for n, i in indices:
    assert out_flat[n][i] == out_expected_flat[n][i]
    assert np.dtype(out_flat[n][i]) == np.dtype(out_expected_flat[n][i])
