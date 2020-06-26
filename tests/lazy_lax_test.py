import itertools
import collections
from functools import partial

import pytest
import numpy as np

from jax import dtypes
import jax.test_util as jtu
from jax import lax

import fastar.test_util as tu

# This is borrowed from lax_tests.py in the JAX tests directory.
# TODO import directly from lax_tests.py

def supported_dtypes(dtypes):
  return [t for t in dtypes if t in jtu.supported_dtypes()]

float_dtypes = supported_dtypes([dtypes.bfloat16, np.float16, np.float32,
                                 np.float64])
complex_elem_dtypes = supported_dtypes([np.float32, np.float64])
complex_dtypes = supported_dtypes([np.complex64, np.complex128])
inexact_dtypes = float_dtypes + complex_dtypes
int_dtypes = supported_dtypes([np.int32, np.int64])
uint_dtypes = supported_dtypes([np.uint32, np.uint64])
bool_dtypes = [np.bool_]
default_dtypes = float_dtypes + int_dtypes
all_dtypes = float_dtypes + complex_dtypes + int_dtypes + bool_dtypes

compatible_shapes = [[(3,)], [(3, 4), (3, 1), (1, 4)], [(2, 3, 4), (2, 1, 4)]]

CombosWithReplacement = itertools.combinations_with_replacement


OpRecord = collections.namedtuple(
    "OpRecord", ["op", "nargs", "dtypes", "rng_factory", "tol"])

def op_record(op, nargs, dtypes, rng_factory, tol=None):
  return OpRecord(op, nargs, dtypes, rng_factory, tol)
LAX_OPS = [
    op_record("neg", 1, default_dtypes + complex_dtypes, jtu.rand_small),
    op_record("sign", 1, default_dtypes + uint_dtypes, jtu.rand_small),
    op_record("floor", 1, float_dtypes, jtu.rand_small),
    op_record("ceil", 1, float_dtypes, jtu.rand_small),
    op_record("round", 1, float_dtypes, jtu.rand_default),
    # TODO (j-towns): requires lax.broadcast_in_dim
    # op_record("nextafter", 2, [f for f in float_dtypes if f != dtypes.bfloat16],
    #           jtu.rand_default, tol=0),

    op_record("is_finite", 1, float_dtypes, jtu.rand_small),

    op_record("exp", 1, float_dtypes + complex_dtypes, jtu.rand_small),
    # TODO(b/142975473): on CPU, expm1 for float64 is only accurate to ~float32
    # precision.
    op_record("expm1", 1, float_dtypes + complex_dtypes, jtu.rand_small,
              {np.float64: 1e-8}),
    op_record("log", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    op_record("log1p", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    # TODO(b/142975473): on CPU, tanh for complex128 is only accurate to
    # ~float32 precision.
    # TODO(b/143135720): on GPU, tanh has only ~float32 precision.
    op_record("tanh", 1, float_dtypes + complex_dtypes, jtu.rand_small,
              {np.float64: 1e-9, np.complex128: 1e-7}),
    op_record("sin", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("cos", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("atan2", 2, float_dtypes, jtu.rand_default),

    op_record("sqrt", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    op_record("rsqrt", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    op_record("square", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("reciprocal", 1, float_dtypes + complex_dtypes, jtu.rand_positive),
    # TODO (j-towns): requires jit
    # op_record("tan", 1, float_dtypes, jtu.rand_default, {np.float32: 3e-5}),
    # TODO (j-towns): requires jit
    # op_record("asin", 1, float_dtypes, jtu.rand_small),
    # TODO (j-towns): requires jit
    # op_record("acos", 1, float_dtypes, jtu.rand_small),
    op_record("atan", 1, float_dtypes, jtu.rand_small),
    op_record("asinh", 1, float_dtypes, jtu.rand_default),
    op_record("acosh", 1, float_dtypes, jtu.rand_positive),
    # TODO(b/155331781): atanh has only ~float precision
    op_record("atanh", 1, float_dtypes, jtu.rand_small, {np.float64: 1e-9}),
    op_record("sinh", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("cosh", 1, float_dtypes + complex_dtypes, jtu.rand_default),
    op_record("lgamma", 1, float_dtypes, jtu.rand_positive,
              {np.float32: 1e-3 if jtu.device_under_test() == "tpu" else 1e-5,
               np.float64: 1e-14}),
    op_record("digamma", 1, float_dtypes, jtu.rand_positive,
              {np.float64: 1e-14}),
    op_record("betainc", 3, float_dtypes, jtu.rand_positive,
              {np.float64: 1e-14}),
    op_record("igamma", 2,
              [f for f in float_dtypes if f not in [dtypes.bfloat16, np.float16]],
              jtu.rand_positive, {np.float64: 1e-14}),
    op_record("igammac", 2,
              [f for f in float_dtypes if f not in [dtypes.bfloat16, np.float16]],
              jtu.rand_positive, {np.float64: 1e-14}),
    op_record("erf", 1, float_dtypes, jtu.rand_small),
    op_record("erfc", 1, float_dtypes, jtu.rand_small),
    # TODO(b/142976030): the approximation of erfinf used by XLA is only
    # accurate to float32 precision.
    op_record("erf_inv", 1, float_dtypes, jtu.rand_small,
              {np.float64: 1e-9}),
    op_record("bessel_i0e", 1, float_dtypes, jtu.rand_default),
    op_record("bessel_i1e", 1, float_dtypes, jtu.rand_default),

    op_record("real", 1, complex_dtypes, jtu.rand_default),
    op_record("imag", 1, complex_dtypes, jtu.rand_default),
    # TODO (j-towns) requires lax.broadcast_in_dim
    # op_record("complex", 2, complex_elem_dtypes, jtu.rand_default),
    op_record("conj", 1, complex_elem_dtypes + complex_dtypes,
              jtu.rand_default),
    op_record("abs", 1, default_dtypes + complex_dtypes, jtu.rand_default),
    op_record("pow", 2, float_dtypes + complex_dtypes, jtu.rand_positive),

    op_record("bitwise_and", 2, bool_dtypes, jtu.rand_small),
    op_record("bitwise_not", 1, bool_dtypes, jtu.rand_small),
    op_record("bitwise_or", 2, bool_dtypes, jtu.rand_small),
    op_record("bitwise_xor", 2, bool_dtypes, jtu.rand_small),
    op_record("population_count", 1, uint_dtypes, partial(jtu.rand_int,
                                                          high=1 << 32)),

    op_record("add", 2, default_dtypes + complex_dtypes, jtu.rand_small),
    op_record("sub", 2, default_dtypes + complex_dtypes, jtu.rand_small),
    op_record("mul", 2, default_dtypes + complex_dtypes, jtu.rand_small),
    op_record("div", 2, default_dtypes + complex_dtypes, jtu.rand_nonzero),
    op_record("rem", 2, default_dtypes, jtu.rand_nonzero),

    op_record("max", 2, all_dtypes, jtu.rand_small),
    op_record("min", 2, all_dtypes, jtu.rand_small),

    op_record("eq", 2, all_dtypes, jtu.rand_some_equal),
    op_record("ne", 2, all_dtypes, jtu.rand_small),
    op_record("ge", 2, default_dtypes, jtu.rand_small),
    op_record("gt", 2, default_dtypes, jtu.rand_small),
    op_record("le", 2, default_dtypes, jtu.rand_small),
    op_record("lt", 2, default_dtypes, jtu.rand_small),
]


@pytest.mark.parametrize(
    'op_name,rng_factory,shapes,dtype,tol',
    [(rec.op, rec.rng_factory, shapes, dtype, rec.tol)
     for rec in LAX_OPS
     for shape_group in compatible_shapes
     for shapes in CombosWithReplacement(shape_group, rec.nargs)
     for dtype in rec.dtypes])
def test_lazy(op_name, rng_factory, shapes, dtype, tol):
  rng = rng_factory(np.random)
  args = [rng(shape, dtype) for shape in shapes]
  tu.check_lazy_fun(getattr(lax, op_name), *args, atol=tol, rtol=tol)

@pytest.mark.parametrize(
    'dim,base_shape,dtype,num_arrs,rng_factory',
    [(dim, base_shape, dtype, num_arrs, rng_factory)
     for num_arrs in [3]
     for dtype in default_dtypes
     for base_shape in [(4,), (3, 4), (2, 3, 4)]
     for dim in range(len(base_shape))
     for rng_factory in [jtu.rand_default]])
def test_concatenate(dim, base_shape, dtype, num_arrs, rng_factory):
  rng = rng_factory(np.random)
  shapes = [base_shape[:dim] + (size,) + base_shape[dim+1:]
            for size, _ in zip(itertools.cycle([3, 1, 4]), range(num_arrs))]
  args = [rng(shape, dtype) for shape in shapes]
  op = lambda *args: lax.concatenate(args, dim)
  tu.check_lazy_fun(op, *args)

@pytest.mark.parametrize(
    'shape,dtype,starts,limits,strides,rng_factory',
    [(shape, dtype, start_indices, limit_indices, strides, rng_factory)
     for shape, start_indices, limit_indices, strides in [
         [(3,), (1,), (2,), None],
         [(7,), (4,), (7,), None],
         [(5,), (1,), (5,), (2,)],
         [(8,), (1,), (6,), (2,)],
         [(5, 3), (1, 1), (3, 2), None],
         [(5, 3), (1, 1), (3, 1), None],
         [(7, 5, 3), (4, 0, 1), (7, 1, 3), None],
         [(5, 3), (1, 1), (2, 1), (1, 1)],
         [(5, 3), (1, 1), (5, 3), (2, 1)],
         ]
     for dtype in default_dtypes
     for rng_factory in [jtu.rand_default]
     if strides is None])
def test_slice(shape, dtype, starts, limits, strides, rng_factory):
  rng = rng_factory(np.random)
  args = [rng(shape, dtype)]
  op = lambda x: lax.slice(x, starts, limits, strides)
  tu.check_lazy_fun(op, *args)
