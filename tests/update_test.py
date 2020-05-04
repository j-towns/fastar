from pytest import mark
import numpy as onp

import jax.scipy.special as special
import numpy as onp
from jax import jit
from jax import lax
from absl.testing import parameterized

from fastar.test_util import check


rng = onp.random.RandomState(0)

randn   =     lambda shape: onp.float32(rng.randn(*shape))
randpos =     lambda shape: onp.float32(onp.abs(rng.randn(*shape)))
randuniform = lambda shape: onp.float32(rng.rand(*shape))
randbool =    lambda shape: rng.randn(*shape) > rng.randn(*shape)


# Unary elementwise operations
def test_abs():   check(lax.abs,       randn((3, 1)))
def test_ceil():  check(lax.ceil,      randn((3, 1)))
def test_cos():   check(lax.cos,       randn((3, 1)))
def test_exp():   check(lax.exp,       randn((3, 1)))
def test_expm1(): check(lax.expm1,     randn((3, 1)))
def test_floor(): check(lax.floor,     randn((3, 1)))
def test_log():   check(lax.log,       randpos((3, 1)))
def test_log1p(): check(lax.log1p,     randpos((3, 1)))
def test_neg():   check(lax.neg,       randn((3, 1)))
def test_sign():  check(lax.sign,      randn((3, 1)))
def test_sin():   check(lax.sin,       randn((3, 1)))
def test_tanh():  check(lax.tanh,      randn((3, 1)))

def test_expit(): check(special.expit, randn((3, 1)))
def test_logit(): check(special.logit, randuniform((3, 1)))

def test_convert(): check(
    lambda arg: lax.convert_element_type(arg, new_dtype=onp.int), randn((3, 1)))

# Binary elementwise operations
def test_add(): check(lax.add, randn((3, 1)), randn((1, 2)))
def test_div(): check(lax.div, randn((3, 1)), randn((1, 2)))
def test_eq():  check(lax.eq,  randn((3, 1)), randn((1, 2)))
def test_ge():  check(lax.ge,  randn((3, 1)), randn((1, 2)))
def test_gt():  check(lax.gt,  randn((3, 1)), randn((1, 2)))
def test_le():  check(lax.le,  randn((3, 1)), randn((1, 2)))
def test_lt():  check(lax.lt,  randn((3, 1)), randn((1, 2)))
def test_max(): check(lax.max, randn((3, 1)), randn((1, 2)))
def test_min(): check(lax.min, randn((3, 1)), randn((1, 2)))
def test_mul(): check(lax.mul, randn((3, 1)), randn((1, 2)))
def test_or():  check(lax.or,  randbool((3, 1)), randbool((1, 2)))
def test_ne():  check(lax.ne,  randn((3, 1)), randn((1, 2)))
def test_rem(): check(lax.rem, randn((3, 1)), randn((1, 2)))
def test_sub(): check(lax.sub, randn((3, 1)), randn((1, 2)))

def test_add_scalar_scalar(): check(lax.add, randn(()), randn(()))
def test_add_scalar_vector(): check(lax.add, randn(()), randn((2,)))
def test_add_scalar_scalar_int(): check(lax.add, 3, 4)
def test_add_vector_vector(): check(lax.add, randn((1,)), randn((2,)))

@mark.parametrize('axes,shape', [((0,), (4,)),
                                 ((0,), (2, 4)),
                                 ((1,), (2, 4)),
                                 ((0, 1), (2, 4)),
                                 ((0, 2), (2, 3, 4))])
def test_reduce_sum(axes, shape):
  check(lambda x: lax._reduce_sum(x, axes=axes), randn((shape)))

def test_reduce_min():
  check(lambda x: lax._reduce_min(x, axes=(0,)), randn((2, 4)))

def test_reduce_max():
  check(lambda x: lax._reduce_max(x, axes=(0,)), randn((2, 4)))

@mark.parametrize('shapes', [((2,), (2,)),
                             ((3, 2), (2,)),
                             ((2,), (2, 3)),
                             ((3, 2), (2, 4))])
def test_dot_vector_vector(shapes):
  shape_a, shape_b = shapes
  check(lax.dot, randn((shape_a)), randn((shape_b)))

def test_dot_general_tensor_matrix():
  check(
      lambda x, y: lax.dot_general(x, y, dimension_numbers=(((2,), (1,)),
                                                            ((0,), (0,)))),
      randn((1, 2, 3)), randn((1, 3)))

def test_dot_general_tensor_tensor():
  check(
      lambda x, y: lax.dot_general(x, y, dimension_numbers=(((2,), (2,)),
                                                            ((1, 0), (1, 0)))),
      randn((1, 2, 3)), randn((1, 2, 3, 4)), rtol=1e-4)

def test_select():
  check(lax.select, randbool((3, 2)), randn((3, 2)), randn((3, 2)))

def test_select_scalar_pred():
  check(lax.select, randbool(()), randn((3, 2)), randn((3, 2)))

def test_transpose():
  check(lambda x: lax.transpose(x, permutation=(1, 2, 0)), randn((1, 2, 3)))

def test_reverse():
  check(lambda x: lax.rev(x, dimensions=(1, 2)), randn((1, 2, 3)))

def test_reshape():
  check(lambda x: lax.reshape(x, new_sizes=(3, 2), dimensions=(1, 0, 2)),
        randn((1, 2, 3)))

def test_concatenate_2():
  check(lambda x, y: lax.concatenate((x, y), dimension=2),
        randn((1, 2, 1)), randn((1, 2, 3)))

def test_concatenate_3():
  check(lambda x, y, z: lax.concatenate((x, y, z), dimension=1),
        randn((2, 1)), randn((2, 3)), randn((2, 2)))

def test_slice():
  check(
    lambda x: lax.slice(
        x, start_indices=(1, 1, 0), limit_indices=(2, 2, 1), strides=(2, 1, 1)),
    randn((7, 2, 1)))

def test_index(): check(lambda x: x[0:2], randn((4,)))

# TODO implement simultaneous negative + interior:
# (((-1, -2, 2),), (5,)), (((-1, -2, 1), (1, 2, 2)), (4, 2)))
@mark.parametrize('padding_config,shape',
                  [(((1, 2, 0),),             (2,)),
                   (((1, 2, 0), (3, 4, 0)),   (1, 2)),
                   (((0, 0, 0), (0, 0, 0)),   (1, 2)),
                   (((1, 2, 3),),             (2,)),
                   (((1, 2, 1), (3, 4, 2)),   (3, 2)),
                   (((-1, 2, 0),),            (2,)),
                   (((-1, -2, 0), (1, 2, 0)), (4, 2)),
                   (((-1, 2, 0), (1, 2, 2)),  (4, 2))])
def test_pad_vector(padding_config, shape):
  check(lambda x, padding_value: lax.pad(x, padding_value,
                                         padding_config=padding_config),
        randn(shape), randn(()))

@mark.parametrize(
    'strides,padding,lhs_dilation,dimension_numbers,lhs_perm,rhs_perm',
    [(strides, padding, lhs_dilation, dimension_numbers, lhs_perm, rhs_perm)
     for strides in ((1, 2), (2, 1))
     for padding in (((0, 0), (0, 0)), 'VALID', 'SAME')
     for lhs_dilation in (None, (1, 2))
     for dimension_numbers, (lhs_perm, rhs_perm) in (
         (("NCHW", "OIHW", "NCHW"), ((0, 1, 2, 3), (0, 1, 2, 3))),
         (("NHWC", "HWIO", "NHWC"), ((0, 2, 3, 1), (2, 3, 1, 0))),
         (("NCHW", "HWIO", "NHWC"), ((0, 1, 2, 3), (2, 3, 1, 0))))
     # String padding is not implemented for transposed convolution, see
     # conv_general_dilated implementation:
     if lhs_dilation is None or not isinstance(padding, str)])
def test_convolution(strides, padding, lhs_dilation, dimension_numbers,
                     lhs_perm, rhs_perm):
  lhs_shape = (1, 2, 4, 4)
  rhs_shape = (3, 2, 3, 2)

  lhs = randn(onp.take(lhs_shape, lhs_perm))
  rhs = randn(onp.take(rhs_shape, rhs_perm))
  check(
    lambda lhs: lax.conv_general_dilated(
      lhs, rhs, strides, padding, lhs_dilation=lhs_dilation,
      dimension_numbers=dimension_numbers),
    lhs, rtol=1e-4, atol=1e-6)

def test_jit():
  check(jit(lambda x: x * 2), randn((1,)))

def test_jit_freevar():
  check(lambda x, y: jit(lambda x: x * y)(x), randn((1,)), randn((1,)))

def test_jit_indexing():
  check(jit(lambda x: x[:2]), randn((4,)))
