import numpy as np

import jax.test_util as jtu
from jax import lax
from jax.util import safe_map, safe_zip
from jax.tree_util import tree_flatten, tree_map
from jax import ShapeDtypeStruct

from fastar import delay, force

map = safe_map
zip = safe_zip


rng = np.random.RandomState(0)

def check(thunk, atol=None, rtol=None):
  expected = thunk()
  expected_shape_dtype = tree_map(
      lambda e: ShapeDtypeStruct(e.shape, e.dtype), expected)
  delayed = delay(thunk, expected_shape_dtype)
  result = force(delayed)
  jtu.check_close(result, expected, atol, rtol)

def random_box(rng, shape):
  if np.any(np.less(shape, 0)):
    raise ValueError
  box_start = rng.randint(np.maximum(shape, 1))
  box_shape = rng.randint(np.maximum(shape - (box_start - 1), 1))
  return box_start, box_shape

def random_box_thunk(rng, thunk):
  out_flat, _ = tree_flatten(thunk())
  idx = rng.randint(len(out_flat))
  box_start, box_shape = random_box(rng, out_flat[idx].shape)
  def new_thunk():
    out_flat, _ = tree_flatten(thunk())
    return lax.slice(out_flat[idx], box_start, np.add(box_start, box_shape))
  return new_thunk

def check_multibox(thunk, n=2, rng=rng, atol=None, rtol=None):
  for _ in range(n):
    check(random_box_thunk(rng, thunk), atol=atol, rtol=rtol)
