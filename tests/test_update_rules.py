import functools

import jax.lax as lax
import numpy as np

from fastar.test_util import check_fun

rng = np.random.RandomState(0)
R = rng.randn
check = functools.partial(check_fun, rng)


# Unops
def test_abs(): check(lax.abs, R(1, 2))
def test_ceil(): check(lax.ceil, R(1, 2))
def test_cos(): check(lax.cos, R(1, 2))
def test_sin(): check(lax.sin, R(1, 2))
def test_exp(): check(lax.exp, R(1, 2))
def test_floor(): check(lax.floor, R(1, 2))
def test_log(): check(lax.log, R(1, 2))
def test_neg(): check(lax.neg, R(1, 2))
def test_sign(): check(lax.sign, R(1, 2))
def test_tanh(): check(lax.tanh, R(1, 2))


# Binops
def test_add_scalar(): check(lax.add, R(), R())
def test_add_scalar_as_numpy(): check(lax.add, np.float64(R()), np.float64(R()))
def test_add_scalar_as_array(): check(lax.add, np.array(R()), np.array(R()))
def test_add_scalar_int(): check(lax.add, 4, 7)
def test_add_vector(): check(lax.add, R(1), R(2))
def test_add_matrix(): check(lax.add, R(1, 2), R(3, 1))


def test_sub(): check(lax.sub, R(1, 2), R(3, 1))
def test_mul(): check(lax.mul, R(1, 2), R(3, 1))
def test_div(): check(lax.div, R(1, 2), R(3, 1))
def test_rem(): check(lax.rem, R(1, 2), R(3, 1))
def test_max(): check(lax.max, R(1, 2), R(3, 1))
def test_min(): check(lax.min, R(1, 2), R(3, 1))


def test_reduce_sum_vector(): check(lambda x: lax._reduce_sum(x, axes=(0,)), R(4))
def test_reduce_sum_matrix_axis0(): check(lambda x: lax._reduce_sum(x, axes=(0,)), R(2, 4))
def test_reduce_sum_matrix_axis1(): check(lambda x: lax._reduce_sum(x, axes=(1,)), R(2, 4))
def test_reduce_sum_matrix_both(): check(lambda x: lax._reduce_sum(x, axes=(0, 1)), R(2, 4))
def test_reduce_sum_tensor(): check(lambda x: lax._reduce_sum(x, axes=(0, 2)), R(2, 3, 4))


def test_reduce_min(): check(lambda x: lax._reduce_min(x, axes=(0,)), R(2, 4))
def test_reduce_max(): check(lambda x: lax._reduce_max(x, axes=(0,)), R(2, 4))


def test_dot_vector_vector(): check(lax.dot, R(2), R(2))
def test_dot_matrix_vector(): check(lax.dot, R(3, 2), R(2))
def test_dot_matrix_matrix(): check(lax.dot, R(3, 2), R(2, 4))


def test_dot_general_tensor_matrix(): check(lambda x, y: lax.dot_general(x, y, dimension_numbers=(((2,), (1,)), ((0,), (0,)))), R(5, 2, 3), R(5, 3))
def test_dot_general_tensor_tensor(): check(lambda x, y: lax.dot_general(x, y, dimension_numbers=(((2,), (1,)), ((0,), (0,)))), R(5, 2, 3), R(5, 3, 4))


def test_transpose(): check(lambda x: lax.transpose(x, permutation=(1, 2, 0)), R(1, 2, 3))
def test_reverse(): check(lambda x: lax.rev(x, dimensions=(1, 2)), R(1, 2, 3))
def test_reshape(): check(lambda x: lax.reshape(x, new_sizes=(3,2), dimensions=(1, 0, 2)), R(1, 2, 3))


def test_pad_vector(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((1,2,0),)), R(2), R())
def test_pad_matrix(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((1,2,0),(3,4,0))), R(1,2), R())
def test_pad_vector_interior(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((1,2,3),)), R(2), R())
def test_pad_matrix_interior(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((1,2,1),(3,4,2))), R(3,2), R())
