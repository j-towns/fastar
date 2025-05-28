import functools
import collections
import itertools

from jax import lax
from jax import dtypes
import jax.numpy as jnp
import numpy as np
import numpy.testing as np_testing
import pytest

from fastar import jax_test_util as jtu
from fastar import test_util
from fastar.core import ScanConversionError
from fastar.util import safe_map, safe_zip

###############################################################################
# This section is copied from jax/_src/internal_test_util/lax_test_util.py

# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

# For standard unops and binops, we can generate a large number of tests on
# arguments of appropriate shapes and dtypes using the following table.

float_dtypes = jtu.dtypes.all_floating
complex_elem_dtypes = jtu.dtypes.floating
complex_dtypes = jtu.dtypes.complex
inexact_dtypes = jtu.dtypes.all_inexact
int_dtypes = jtu.dtypes.all_integer
uint_dtypes = jtu.dtypes.all_unsigned
bool_dtypes = jtu.dtypes.boolean

default_dtypes = float_dtypes + int_dtypes
number_dtypes = (
    float_dtypes + complex_dtypes + int_dtypes + uint_dtypes
)
all_dtypes = (
    number_dtypes + bool_dtypes
)
python_scalar_types = [bool, int, float, complex]

compatible_shapes = [[(3,)], [(2, 3, 4), (2, 1, 4)], [(1, 1), (1, 3)]]

OpRecord = collections.namedtuple(
    "OpRecord", ["op", "nargs", "dtypes", "rng_factory", "tol"]
)


def op_record(op, nargs, dtypes, rng_factory, tol=None):
  return OpRecord(op, nargs, dtypes, rng_factory, tol)


NamedReducerOpRecord = collections.namedtuple(
    "NamedReducerOpRecord", ["op", "reference_op", "dtypes"]
)

def lax_named_reduce_ops():
  return [
      NamedReducerOpRecord(lax.reduce_sum, np.sum, number_dtypes),
      NamedReducerOpRecord(lax.reduce_prod, np.prod, number_dtypes),
      NamedReducerOpRecord(lax.reduce_max, np.max, all_dtypes),
      NamedReducerOpRecord(lax.reduce_min, np.min, all_dtypes),
      NamedReducerOpRecord(lax.reduce_and, np.bitwise_and.reduce,
                           bool_dtypes + int_dtypes + uint_dtypes),
      NamedReducerOpRecord(lax.reduce_or, np.bitwise_or.reduce,
                           bool_dtypes + int_dtypes + uint_dtypes),
      NamedReducerOpRecord(lax.reduce_xor, np.bitwise_xor.reduce,
                           bool_dtypes + int_dtypes + uint_dtypes),
  ]


def lax_ops():
  return [
      op_record(
          "neg", 1, default_dtypes + complex_dtypes, jtu.rand_small
      ),
      op_record("sign", 1, default_dtypes + uint_dtypes, jtu.rand_small),
      op_record("floor", 1, float_dtypes, jtu.rand_small),
      op_record("ceil", 1, float_dtypes, jtu.rand_small),
      op_record("round", 1, float_dtypes, jtu.rand_default),
      op_record(
          "nextafter",
          2,
          [f for f in float_dtypes if f != dtypes.bfloat16],
          jtu.rand_default,
          tol=0,
      ),
      op_record("is_finite", 1, float_dtypes, jtu.rand_small),
      op_record("exp", 1, float_dtypes + complex_dtypes, jtu.rand_small),
      op_record("exp2", 1, float_dtypes + complex_dtypes, jtu.rand_small),
      # TODO(b/142975473): on CPU, expm1 for float64 is only accurate to ~float32
      # precision.
      op_record(
          "expm1",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_small,
          {np.float64: 1e-8},
      ),
      op_record(
          "log", 1, float_dtypes + complex_dtypes, jtu.rand_positive
      ),
      op_record(
          "log1p", 1, float_dtypes + complex_dtypes, jtu.rand_positive
      ),
      # TODO(b/142975473): on CPU, tanh for complex128 is only accurate to
      # ~float32 precision.
      # TODO(b/143135720): on GPU, tanh has only ~float32 precision.
      op_record(
          "tanh",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_small,
          {np.float64: 1e-9, np.complex128: 1e-7},
      ),
      op_record(
          "logistic", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "sin", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "cos", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record("atan2", 2, float_dtypes, jtu.rand_default),
      op_record("sqrt", 1, float_dtypes, jtu.rand_positive),
      op_record("sqrt", 1, complex_dtypes, jtu.rand_default),
      op_record("rsqrt", 1, float_dtypes, jtu.rand_positive),
      op_record("rsqrt", 1, complex_dtypes, jtu.rand_default),
      op_record("cbrt", 1, float_dtypes, jtu.rand_default),
      op_record(
          "square", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "reciprocal",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_positive,
      ),
      op_record(
          "tan",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_default,
          {np.float32: 3e-5},
      ),
      op_record(
          "asin",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_small,
          {np.complex128: 5e-12},
      ),
      op_record("acos", 1, float_dtypes + complex_dtypes, jtu.rand_small),
      op_record("atan", 1, float_dtypes + complex_dtypes, jtu.rand_small),
      op_record(
          "asinh",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_default,
          tol={np.complex64: 1e-4, np.complex128: 1e-5},
      ),
      op_record(
          "acosh", 1, float_dtypes + complex_dtypes, jtu.rand_positive
      ),
      # TODO(b/155331781): atanh has only ~float precision
      op_record(
          "atanh",
          1,
          float_dtypes + complex_dtypes,
          jtu.rand_small,
          {np.float64: 1e-9},
      ),
      op_record(
          "sinh", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "cosh", 1, float_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "lgamma",
          1,
          float_dtypes,
          jtu.rand_positive,
          {
              np.float32: 1e-5,
              np.float64: 1e-14,
          },
      ),
      op_record(
          "digamma",
          1,
          float_dtypes,
          jtu.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record(
          "betainc",
          3,
          float_dtypes,
          jtu.rand_uniform,
          {
              np.float32: 1e-5,
              np.float64: 1e-12,
          },
      ),
      op_record(
          "igamma",
          2,
          [f for f in float_dtypes if f not in [dtypes.bfloat16, np.float16]],
          jtu.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record(
          "igammac",
          2,
          [f for f in float_dtypes if f not in [dtypes.bfloat16, np.float16]],
          jtu.rand_positive,
          {np.float64: 1e-14},
      ),
      op_record("erf", 1, float_dtypes, jtu.rand_small),
      op_record("erfc", 1, float_dtypes, jtu.rand_small),
      # TODO(b/142976030): the approximation of erfinf used by XLA is only
      # accurate to float32 precision.
      op_record(
          "erf_inv", 1, float_dtypes, jtu.rand_small, {np.float64: 1e-9}
      ),
      op_record("bessel_i0e", 1, float_dtypes, jtu.rand_default),
      op_record("bessel_i1e", 1, float_dtypes, jtu.rand_default),
      op_record("real", 1, complex_dtypes, jtu.rand_default),
      op_record("imag", 1, complex_dtypes, jtu.rand_default),
      op_record("complex", 2, complex_elem_dtypes, jtu.rand_default),
      op_record(
          "conj",
          1,
          complex_elem_dtypes + complex_dtypes,
          jtu.rand_default,
      ),
      op_record(
          "abs", 1, default_dtypes + complex_dtypes, jtu.rand_default
      ),
      op_record(
          "pow", 2, float_dtypes + complex_dtypes, jtu.rand_positive
      ),
      op_record("bitwise_and", 2, bool_dtypes, jtu.rand_small),
      op_record("bitwise_not", 1, bool_dtypes, jtu.rand_small),
      op_record("bitwise_or", 2, bool_dtypes, jtu.rand_small),
      op_record("bitwise_xor", 2, bool_dtypes, jtu.rand_small),
      op_record(
          "population_count", 1, int_dtypes + uint_dtypes, jtu.rand_int
      ),
      op_record("clz", 1, int_dtypes + uint_dtypes, jtu.rand_int),
      op_record(
          "add", 2, default_dtypes + complex_dtypes, jtu.rand_small
      ),
      op_record(
          "sub", 2, default_dtypes + complex_dtypes, jtu.rand_small
      ),
      op_record(
          "mul", 2, default_dtypes + complex_dtypes, jtu.rand_small
      ),
      op_record(
          "div", 2, default_dtypes + complex_dtypes, jtu.rand_nonzero
      ),
      op_record("rem", 2, default_dtypes, jtu.rand_nonzero),
      op_record("max", 2, all_dtypes, jtu.rand_small),
      op_record("min", 2, all_dtypes, jtu.rand_small),
      op_record("eq", 2, all_dtypes, jtu.rand_some_equal),
      op_record("ne", 2, all_dtypes, jtu.rand_small),
      op_record("ge", 2, default_dtypes, jtu.rand_small),
      op_record("gt", 2, default_dtypes, jtu.rand_small),
      op_record("le", 2, default_dtypes, jtu.rand_small),
      op_record("lt", 2, default_dtypes, jtu.rand_small),
      op_record("polygamma", 2, float_dtypes, jtu.rand_positive),
      op_record("zeta", 2, float_dtypes, jtu.rand_positive),
  ]

###############################################################################

@pytest.mark.parametrize(
    'op_name,argnum,rng_factory,shapes,dtype,tol',
    [(rec.op, argnum, rec.rng_factory, shapes, dtype, rec.tol)
     for rec in lax_ops()
     for shape_group in compatible_shapes
     for shapes in itertools.combinations_with_replacement(
         shape_group, rec.nargs
     )
     for dtype in rec.dtypes
     for argnum in range(rec.nargs)])
def test_nary(op_name, argnum, rng_factory, shapes, dtype, tol):
    if shapes[argnum][0] == 1 and any(s[0] > 1 for s in shapes):
        return
    rng = rng_factory(np.random)
    args = tuple(rng(shape, dtype) for shape in shapes)
    def f(xs):
        args_ = list(args)
        args_[argnum] = xs
        return getattr(lax, op_name)(*args_)
    test_util.check_scan(f, args[argnum], atol=tol, rtol=tol)

def test_nary_other_axis():
    rng = np.random.RandomState(0)
    xs = rng.randn(2, 3, 4)
    y = rng.randn(3, 2, 4)
    def f(xs):
        return jnp.moveaxis(lax.add(jnp.moveaxis(xs, 0, 1), y), 1, 0)
    test_util.check_scan(f, xs)

@pytest.mark.parametrize(
    'op,shape,axes,dtype',
    [(rec.op, shape, axes, dtype)
     for rec in lax_named_reduce_ops()
     for (shape, axes) in [[(3, 4, 5), (1,)], [(3, 4, 5), (1, 2)]]
     for dtype in rec.dtypes])
def test_reduce_named(op, shape, axes, dtype):
    rng_factory = (jtu.rand_default if dtypes.issubdtype(dtype, np.integer)
                   else jtu.rand_small)
    rng = rng_factory(np.random)
    arg = rng(shape, dtype)
    fun = functools.partial(op, axes=axes)
    test_util.check_scan(fun, arg)

def test_scan():
    rng = np.random.RandomState(0)
    init_carry = np.zeros(2)
    def f(xs):
        carry_out, ys = lax.scan(
            lambda carry, x: (carry + x, carry + x), init_carry, xs
        )
        return ys
    xs = rng.randn(5, 2)
    test_util.check_scan(f, xs)

def test_transpose():
    rng = np.random.RandomState(0)
    def f(xs):
        return lax.transpose(lax.transpose(xs, (1, 2, 0)), (2, 1, 0))
    xs = rng.randn(2, 3, 4)
    test_util.check_scan(f, xs)

def test_transpose_wrong_axis():
    rng = np.random.RandomState(0)
    def f(xs):
        return lax.transpose(xs, (1, 2, 0))
    xs = rng.randn(2, 3, 4)
    np_testing.assert_raises(ScanConversionError, test_util.check_scan, f, xs)

def test_broadcast_in_dim():
    rng = np.random.RandomState(0)
    def f(xs):
        return lax.broadcast_in_dim(xs, (2, 3, 4, 5), (0, 1, 2))
    xs = rng.randn(2, 1, 4)
    test_util.check_scan(f, xs)

def test_broadcast_in_dim_other_axis():
    rng = np.random.RandomState(0)
    def f(xs):
        xs = jnp.moveaxis(xs, 0, 1)
        return jnp.moveaxis(
            lax.broadcast_in_dim(xs, (3, 2, 4, 5), (0, 1, 2)), 1, 0
        )
    xs = rng.randn(2, 1, 4)
    test_util.check_scan(f, xs)

def test_conv_batch():
    rng = np.random.RandomState(0)
    lhs = rng.randn(2, 3, 4, 5)
    rhs = rng.randn(1, 2, 5, 6)
    def f(x):
        return lax.conv_general_dilated(
            x, rhs, window_strides=[1, 1], padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
    test_util.check_scan(f, lhs)

def test_conv_causal():
    window_size = 2
    rng = np.random.RandomState(0)
    lhs = rng.randn(6, 4, 5)
    rhs = rng.randn(window_size, 5, 6)
    def f(x):
        return lax.conv_general_dilated(
            x, rhs, window_strides=[1], padding=[(window_size - 1, 0)],
            dimension_numbers=("TNC", "TIO", "TNC"),
        )
    test_util.check_scan(f, lhs)

def test_slice():
    rng = np.random.RandomState(0)
    operand = rng.randn(6, 4, 5)
    def f(operand):
        return lax.slice(operand, (0, 2, 1), (6, 3, 5), (1, 1, 2))
    test_util.check_scan(f, operand)
