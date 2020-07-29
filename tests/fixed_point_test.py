import fastar.test_util as tu
from fastar import lazy_eval_fixed_point

import jax.numpy as np
import jax.test_util as jtu
from jax import lax
import jax


def test_fixedpoint_simple():
  def fixed_point(x):
    return np.concatenate([np.array([1.]), 2 * lax.slice(x, [0], [3])])

  x_mock = np.zeros(4)
  tu.check_lazy_fixed_point(fixed_point, x_mock)

def test_fixedpoint_2d():
  def fixed_point(x):
    first_row = np.concatenate(
        [np.array([[1.]]), 2 * lax.slice(x, [0, 0], [1, 3])], -1)
    second_row = np.concatenate([lax.slice(first_row, [0, 2], [1, 4]),
                                 np.array([[3., 4]])], -1)
    return np.concatenate([first_row, second_row], 0)

  x_mock = np.zeros((2, 4))
  tu.check_lazy_fixed_point(fixed_point, x_mock)

def test_fixedpoint_vmap():
  def elem(y):
    def fixed_point(x):
      return np.concatenate([np.array([1.]), 2 * lax.slice(x + y, [0], [3])])
    return lazy_eval_fixed_point(fixed_point, np.zeros(4))[:]

  ys = np.array([1, 2, 3, 4])
  expected = np.array([elem(y) for y in ys])
  actual = jax.vmap(elem)(ys)
  jtu.check_close(expected, actual)
