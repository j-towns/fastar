import jax.lax as lax
import numpy as np
import pytest

from fastar.test_util import check, check_custom_input


# Unops
def test_abs(): check(lax.abs, (1, 2))
def test_ceil(): check(lax.ceil, (1, 2))
def test_cos(): check(lax.cos, (1, 2))
def test_sin(): check(lax.sin, (1, 2))
def test_exp(): check(lax.exp, (1, 2))
def test_floor(): check(lax.floor, (1, 2))
def test_log(): check_custom_input(lax.log, lambda rng: np.abs(rng.randn(1, 2)))
def test_neg(): check(lax.neg, (1, 2))
def test_sign(): check(lax.sign, (1, 2))
def test_tanh(): check(lax.tanh, (1, 2))


# Binops
def test_add_scalar(): check(lax.add, (), ())
def test_add_scalar_as_numpy(): check_custom_input(lax.add, lambda rng: (np.float64(rng.randn()), np.float64(rng.randn())))
def test_add_scalar_as_array(): check_custom_input(lax.add, lambda rng: (np.array(rng.randn()), np.array(rng.randn())))
def test_add_scalar_int(): check_custom_input(lax.add, lambda _: (4, 7))
def test_add_vector(): check(lax.add, (1,), (2,))
def test_add_matrix(): check(lax.add, (1, 2), (3, 1))


def test_sub(): check(lax.sub, (1, 2), (3, 1))
def test_mul(): check(lax.mul, (1, 2), (3, 1))
def test_div(): check(lax.div, (1, 2), (3, 1))
def test_rem(): check(lax.rem, (1, 2), (3, 1))
def test_max(): check(lax.max, (1, 2), (3, 1))
def test_min(): check(lax.min, (1, 2), (3, 1))


def test_reduce_sum_vector(): check(lambda x: lax._reduce_sum(x, axes=(0,)), (4,))
def test_reduce_sum_matrix_axis0(): check(lambda x: lax._reduce_sum(x, axes=(0,)), (2, 4))
def test_reduce_sum_matrix_axis1(): check(lambda x: lax._reduce_sum(x, axes=(1,)), (2, 4))
def test_reduce_sum_matrix_both(): check(lambda x: lax._reduce_sum(x, axes=(0, 1)), (2, 4))
def test_reduce_sum_tensor(): check(lambda x: lax._reduce_sum(x, axes=(0, 2)), (2, 3, 4))


def test_reduce_min(): check(lambda x: lax._reduce_min(x, axes=(0,)), (2, 4))
def test_reduce_max(): check(lambda x: lax._reduce_max(x, axes=(0,)), (2, 4))


def test_dot_vector_vector(): check(lax.dot, (2,), (2,))
def test_dot_matrix_vector(): check(lax.dot, (3, 2), (2,))
def test_dot_matrix_matrix(): check(lax.dot, (3, 2), (2, 4))


def test_dot_general_tensor_matrix(): check(lambda x, y: lax.dot_general(x, y, dimension_numbers=(((2,), (1,)), ((0,), (0,)))), (1, 2, 3), (1, 3))
def test_dot_general_tensor_tensor(): check(lambda x, y: lax.dot_general(x, y, dimension_numbers=(((2,), (1,)), ((0,), (0,)))), (1, 2, 3), (1, 3, 4), rtol=1e-4)


def test_transpose(): check(lambda x: lax.transpose(x, permutation=(1, 2, 0)), (1, 2, 3))
def test_reverse(): check(lambda x: lax.rev(x, dimensions=(1, 2)), (1, 2, 3))
def test_reshape(): check(lambda x: lax.reshape(x, new_sizes=(3, 2), dimensions=(1, 0, 2)), (1, 2, 3))
def test_concatenate_2(): check(lambda x, y: lax.concatenate((x, y), dimension=2), (1, 2, 1), (1, 2, 3))
def test_concatenate_3(): check(lambda x, y, z: lax.concatenate((x, y, z), dimension=1), (2, 1), (2, 3), (2, 2))

def test_pad_vector(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((1, 2, 0),)), (2,), ())
def test_pad_matrix(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((1, 2, 0), (3, 4, 0))), (1, 2), ())
def test_pad_matrix_zeros(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((0, 0, 0), (0, 0, 0))), (1, 2), ())
def test_pad_vector_interior(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((1, 2, 3),)), (2,), ())
def test_pad_matrix_interior(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((1, 2, 1), (3, 4, 2))), (3, 2), ())
def test_pad_vector_negatively(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((-1, 2, 0),)), (2,), ())
def test_pad_matrix_negatively(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((-1, -2, 0), (1, 2, 0))), (4, 2), ())
def test_pad_matrix_negatively_interior_on_different_axes(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((-1, 2, 0), (1, 2, 2))), (4, 2), ())
#TODO:
#def test_pad_vector_negatively_interior(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((-1, -2, 2),)), (5,), ())
#def test_pad_matrix_negatively_interior(): check(lambda x, padding_value: lax.pad(x, padding_value, padding_config=((-1, -2, 1), (1, 2, 2))), (4, 2), ())

@pytest.mark.parametrize('filter_shape', [(3, 2), (1, 1)])
@pytest.mark.parametrize('strides', [[1, 1], [1, 2], [2, 1]])
@pytest.mark.parametrize('padding', ['SAME', 'VALID'])
@pytest.mark.parametrize('dimension_numbers', [
        (("NCHW", "OIHW", "NCHW"), ([0, 1, 2, 3], [0, 1, 2, 3])),
        (("NHWC", "HWIO", "NHWC"), ([0, 2, 3, 1], [2, 3, 1, 0])),
        (("NCHW", "HWIO", "NHWC"), ([0, 1, 2, 3], [2, 3, 1, 0]))
    ])
def test_convolution(filter_shape, strides, padding, dimension_numbers):
    lhs_shape = (1, 2, 4, 4)
    rhs_shape = (3, 2) + filter_shape
    dimension_numbers, (lhs_perm, rhs_perm) = dimension_numbers
    check(
        lambda lhs: lax.conv_general_dilated(
            lhs, np.random.RandomState(0).randn(*np.take(rhs_shape, rhs_perm)),
            strides, padding, dimension_numbers=dimension_numbers),
        np.take(lhs_shape, lhs_perm))
