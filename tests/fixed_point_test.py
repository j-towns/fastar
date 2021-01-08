import numpy as np

from jax import test_util as jtu, numpy as jnp, lax, vmap, ShapeDtypeStruct

import fastar.test_util as tu
from fastar import delay, force

def naive_eval_fixed_point(fp, init_val):
  val = init_val
  new_val = fp(val)
  while not jnp.all(new_val == val):
    new_val, val = fp(new_val), new_val
  return new_val

def check_fixed_point(fp, x_init):
  def thunk():
    return fp(force(x))
  x = delay(thunk, ShapeDtypeStruct(x_init.shape, x_init.dtype))
  jtu.check_close(naive_eval_fixed_point(fp, x_init), force(x))

def test_fixedpoint_simple():
  def fixed_point(x):
    return jnp.concatenate([jnp.array([1.]), 2 * lax.slice(x, [0], [3])])

  check_fixed_point(fixed_point, jnp.zeros(4))

def test_fixedpoint_2d():
  def fixed_point(x):
    first_row = jnp.concatenate(
        [jnp.array([[1.]]), 2 * lax.slice(x, [0, 0], [1, 3])], -1)
    second_row = jnp.concatenate([lax.slice(first_row, [0, 2], [1, 4]),
                                  jnp.array([[3., 4]])], -1)
    return jnp.concatenate([first_row, second_row], 0)
  check_fixed_point(fixed_point, jnp.zeros((2, 4)))

def test_fixedpoint_vmap():
  def elem(y):
    def fixed_point(x):
      return jnp.concatenate([jnp.array([1.]), 2 * lax.slice(lax.add(x, 2 * y), 
                                                             [0], [3])])
    def thunk():
      return fixed_point(force(x))
    x = delay(thunk, ShapeDtypeStruct((4,), jnp.float32))
    return force(x)

  ys = jnp.array([1., 2., 3., 4.])
  expected = jnp.array([elem(ys[i]) for i in range(4)])
  actual = vmap(elem)(ys)
  jtu.check_close(expected, actual)
