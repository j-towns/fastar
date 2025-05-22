# Contents of this file are copied, with slight modification, from
# jax/_src/test_util.py

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
from functools import partial
import math
import re

import numpy as np

from jax import dtypes as _dtypes
import jax
from jax import config


class _cached_property:
  null = object()

  def __init__(self, method):
    self._method = method
    self._value = self.null

  def __get__(self, obj, cls):
    if self._value is self.null:
      self._value = self._method(obj)
    return self._value

def device_under_test():
  return jax.default_backend()

def supported_dtypes():
  if device_under_test() == "tpu":
    types = {np.bool_, np.int8, np.int16, np.int32, np.uint8, np.uint16,
             np.uint32, _dtypes.bfloat16, np.float16, np.float32, np.complex64,
             _dtypes.float8_e4m3fn, _dtypes.float8_e4m3b11fnuz,
             _dtypes.float8_e5m2}
  elif device_under_test() == "gpu":
    types = {np.bool_, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64,
             _dtypes.bfloat16, np.float16, np.float32, np.float64,
             np.complex64, np.complex128, _dtypes.float8_e4m3fn,
             _dtypes.float8_e5m2}
  elif device_under_test() == "METAL":
    types = {np.int32, np.uint32, np.float32}
  else:
    types = {np.bool_, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64,
             _dtypes.bfloat16, np.float16, np.float32, np.float64,
             np.complex64, np.complex128}
  if not config.jax_enable_x64:
    types -= {np.uint64, np.int64, np.float64, np.complex128}
  return types

class _LazyDtypes:
  """A class that unifies lists of supported dtypes.

  These could be module-level constants, but device_under_test() is not always
  known at import time, so we need to define these lists lazily.
  """
  def supported(self, dtypes):
    supported = supported_dtypes()
    return type(dtypes)(d for d in dtypes if d in supported)

  @_cached_property
  def custom_floats(self):
    float_dtypes = [
      _dtypes.bfloat16,
      _dtypes.float8_e4m3b11fnuz,
      _dtypes.float8_e4m3fn,
      _dtypes.float8_e4m3fnuz,
      _dtypes.float8_e5m2,
      _dtypes.float8_e5m2fnuz,
    ]
    if _dtypes.float8_e3m4 is not None:
      float_dtypes += [_dtypes.float8_e3m4]
    if _dtypes.float8_e4m3 is not None:
      float_dtypes += [_dtypes.float8_e4m3]
    if _dtypes.float8_e8m0fnu is not None:
      float_dtypes += [_dtypes.float8_e8m0fnu]
    return self.supported(float_dtypes)

  @_cached_property
  def floating(self):
    return self.supported([np.float32, np.float64])

  @_cached_property
  def all_floating(self):
    return self.supported([_dtypes.bfloat16, np.float16, np.float32, np.float64])

  @_cached_property
  def integer(self):
    return self.supported([np.int32, np.int64])

  @_cached_property
  def all_integer(self):
    return self.supported([np.int8, np.int16, np.int32, np.int64])

  @_cached_property
  def unsigned(self):
    return self.supported([np.uint32, np.uint64])

  @_cached_property
  def all_unsigned(self):
    return self.supported([np.uint8, np.uint16, np.uint32, np.uint64])

  @_cached_property
  def complex(self):
    return self.supported([np.complex64, np.complex128])

  @_cached_property
  def boolean(self):
    return self.supported([np.bool_])

  @_cached_property
  def inexact(self):
    return self.floating + self.complex

  @_cached_property
  def all_inexact(self):
    return self.all_floating + self.complex

  @_cached_property
  def numeric(self):
    return self.floating + self.integer + self.unsigned + self.complex

  @_cached_property
  def all(self):
    return (self.all_floating + self.all_integer + self.all_unsigned +
            self.complex + self.boolean)

dtypes = _LazyDtypes()

# We use special symbols, represented as singleton objects, to distinguish
# between NumPy scalars, Python scalars, and 0-D arrays.
class ScalarShape:
  def __len__(self): return 0
  def __getitem__(self, i): raise IndexError(f"index {i} out of range.")
class _NumpyScalar(ScalarShape): pass
class _PythonScalar(ScalarShape): pass
NUMPY_SCALAR_SHAPE = _NumpyScalar()
PYTHON_SCALAR_SHAPE = _PythonScalar()


def _dims_of_shape(shape):
  """Converts `shape` to a tuple of dimensions."""
  if type(shape) in (list, tuple):
    return shape
  elif isinstance(shape, ScalarShape):
    return ()
  elif np.ndim(shape) == 0:
    return (shape,)
  else:
    raise TypeError(type(shape))


def _cast_to_shape(value, shape, dtype):
  """Casts `value` to the correct Python type for `shape` and `dtype`."""
  if shape is NUMPY_SCALAR_SHAPE:
    # explicitly cast to NumPy scalar in case `value` is a Python scalar.
    return np.dtype(dtype).type(value)
  elif shape is PYTHON_SCALAR_SHAPE:
    # explicitly cast to Python scalar via https://stackoverflow.com/a/11389998
    return np.asarray(value).item()
  elif type(shape) in (list, tuple):
    assert np.shape(value) == tuple(shape)
    return value
  elif np.ndim(shape) == 0:
    assert np.shape(value) == (shape,)
    return value
  else:
    raise TypeError(type(shape))


def _rand_dtype(rand, shape, dtype, scale=1., post=lambda x: x):
  """Produce random values given shape, dtype, scale, and post-processor.

  Args:
    rand: a function for producing random values of a given shape, e.g. a
      bound version of either np.RandomState.randn or np.RandomState.rand.
    shape: a shape value as a tuple of positive integers.
    dtype: a numpy dtype.
    scale: optional, a multiplicative scale for the random values (default 1).
    post: optional, a callable for post-processing the random values (default
      identity).

  Returns:
    An ndarray of the given shape and dtype using random values based on a call
    to rand but scaled, converted to the appropriate dtype, and post-processed.
  """
  if _dtypes.issubdtype(dtype, np.unsignedinteger):
    r = lambda: np.asarray(scale * abs(rand(*_dims_of_shape(shape)))).astype(dtype)
  else:
    r = lambda: np.asarray(scale * rand(*_dims_of_shape(shape))).astype(dtype)
  if _dtypes.issubdtype(dtype, np.complexfloating):
    vals = r() + 1.0j * r()
  else:
    vals = r()
  return _cast_to_shape(np.asarray(post(vals), dtype), shape, dtype)


def rand_fullrange(rng, standardize_nans=False):
  """Random numbers that span the full range of available bits."""
  def gen(shape, dtype, post=lambda x: x):
    dtype = np.dtype(dtype)
    size = dtype.itemsize * math.prod(_dims_of_shape(shape))
    vals = rng.randint(0, np.iinfo(np.uint8).max, size=size, dtype=np.uint8)
    vals = post(vals).view(dtype)
    if shape is PYTHON_SCALAR_SHAPE:
      # Sampling from the full range of the largest available uint type
      # leads to overflows in this case; sample from signed ints instead.
      if dtype == np.uint64:
        vals = vals.astype(np.int64)
      elif dtype == np.uint32 and not config.enable_x64.value:
        vals = vals.astype(np.int32)
    vals = vals.reshape(shape)
    # Non-standard NaNs cause errors in numpy equality assertions.
    if standardize_nans and np.issubdtype(dtype, np.floating):
      vals[np.isnan(vals)] = np.nan
    return _cast_to_shape(vals, shape, dtype)
  return gen


def rand_default(rng, scale=3):
  return partial(_rand_dtype, rng.randn, scale=scale)


def rand_nonzero(rng):
  post = lambda x: np.where(x == 0, np.array(1, dtype=x.dtype), x)
  return partial(_rand_dtype, rng.randn, scale=3, post=post)


def rand_positive(rng):
  post = lambda x: x + 1
  return partial(_rand_dtype, rng.rand, scale=2, post=post)


def rand_small(rng):
  return partial(_rand_dtype, rng.randn, scale=1e-3)


def rand_not_small(rng, offset=10.):
  post = lambda x: x + np.where(x > 0, offset, -offset)
  return partial(_rand_dtype, rng.randn, scale=3., post=post)


def rand_small_positive(rng):
  return partial(_rand_dtype, rng.rand, scale=2e-5)

def rand_uniform(rng, low=0.0, high=1.0):
  assert low < high
  post = lambda x: x * (high - low) + low
  return partial(_rand_dtype, rng.rand, post=post)


def rand_some_equal(rng):

  def post(x):
    x_ravel = x.ravel()
    if len(x_ravel) == 0:
      return x
    flips = rng.rand(*np.shape(x)) < 0.5
    return np.where(flips, x_ravel[0], x)

  return partial(_rand_dtype, rng.randn, scale=100., post=post)


def rand_some_inf(rng):
  """Return a random sampler that produces infinities in floating types."""
  base_rand = rand_default(rng)

  # TODO: Complex numbers are not correctly tested
  # If blocks should be switched in order, and relevant tests should be fixed
  def rand(shape, dtype):
    """The random sampler function."""
    if not _dtypes.issubdtype(dtype, np.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    if _dtypes.issubdtype(dtype, np.complexfloating):
      base_dtype = np.real(np.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             np.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = np.where(posinf_flips, np.array(np.inf, dtype=dtype), vals)
    vals = np.where(neginf_flips, np.array(-np.inf, dtype=dtype), vals)

    return _cast_to_shape(np.asarray(vals, dtype=dtype), shape, dtype)

  return rand

def rand_some_nan(rng):
  """Return a random sampler that produces nans in floating types."""
  base_rand = rand_default(rng)

  def rand(shape, dtype):
    """The random sampler function."""
    if _dtypes.issubdtype(dtype, np.complexfloating):
      base_dtype = np.real(np.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             np.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    if not _dtypes.issubdtype(dtype, np.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    dims = _dims_of_shape(shape)
    r = rng.rand(*dims)
    nan_flips = r < 0.1
    neg_nan_flips = r < 0.05

    vals = base_rand(shape, dtype)
    vals = np.where(nan_flips, np.array(np.nan, dtype=dtype), vals)
    vals = np.where(neg_nan_flips, np.array(-np.nan, dtype=dtype), vals)

    return _cast_to_shape(np.asarray(vals, dtype=dtype), shape, dtype)

  return rand

def rand_some_inf_and_nan(rng):
  """Return a random sampler that produces infinities in floating types."""
  base_rand = rand_default(rng)

  # TODO: Complex numbers are not correctly tested
  # If blocks should be switched in order, and relevant tests should be fixed
  def rand(shape, dtype):
    """The random sampler function."""
    if not _dtypes.issubdtype(dtype, np.floating):
      # only float types have inf
      return base_rand(shape, dtype)

    if _dtypes.issubdtype(dtype, np.complexfloating):
      base_dtype = np.real(np.array(0, dtype=dtype)).dtype
      out = (rand(shape, base_dtype) +
             np.array(1j, dtype) * rand(shape, base_dtype))
      return _cast_to_shape(out, shape, dtype)

    dims = _dims_of_shape(shape)
    posinf_flips = rng.rand(*dims) < 0.1
    neginf_flips = rng.rand(*dims) < 0.1
    nan_flips = rng.rand(*dims) < 0.1

    vals = base_rand(shape, dtype)
    vals = np.where(posinf_flips, np.array(np.inf, dtype=dtype), vals)
    vals = np.where(neginf_flips, np.array(-np.inf, dtype=dtype), vals)
    vals = np.where(nan_flips, np.array(np.nan, dtype=dtype), vals)

    return _cast_to_shape(np.asarray(vals, dtype=dtype), shape, dtype)

  return rand

# TODO(mattjj): doesn't handle complex types
def rand_some_zero(rng):
  """Return a random sampler that produces some zeros."""
  base_rand = rand_default(rng)

  def rand(shape, dtype):
    """The random sampler function."""
    dims = _dims_of_shape(shape)
    zeros = rng.rand(*dims) < 0.5

    vals = base_rand(shape, dtype)
    vals = np.where(zeros, np.array(0, dtype=dtype), vals)

    return _cast_to_shape(np.asarray(vals, dtype=dtype), shape, dtype)

  return rand


def rand_int(rng, low=0, high=None):
  def fn(shape, dtype):
    nonlocal high
    gen_dtype = dtype if np.issubdtype(dtype, np.integer) else np.int64
    if low == 0 and high is None:
      if np.issubdtype(dtype, np.integer):
        high = np.iinfo(dtype).max
      else:
        raise ValueError("rand_int requires an explicit `high` value for "
                         "non-integer types.")
    return rng.randint(low, high=high, size=shape,
                       dtype=gen_dtype).astype(dtype)
  return fn

def rand_unique_int(rng, high=None):
  def fn(shape, dtype):
    return rng.choice(np.arange(high or math.prod(shape), dtype=dtype),
                      size=shape, replace=False)
  return fn

def rand_indices_unique_along_axis(rng):
  """Sample an array of given shape containing indices up to dim (exclusive),
  such that the indices are unique along the given axis.
  Optionally, convert some of the resulting indices to negative indices."""
  def fn(dim, shape, axis, allow_negative=True):
    batch_size = math.prod(shape[:axis] + shape[axis:][1:])
    idx = [
      rng.choice(dim, size=shape[axis], replace=False)
      for _ in range(batch_size)
    ]
    idx = np.array(idx).reshape(batch_size, shape[axis])
    idx = idx.reshape(shape[:axis] + shape[axis:][1:] + (shape[axis],))
    idx = np.moveaxis(idx, -1, axis)

    # assert that indices are unique along the given axis
    count = partial(np.bincount, minlength=dim)
    assert (np.apply_along_axis(count, axis, idx) <= 1).all()

    if allow_negative:
      mask = rng.choice([False, True], idx.shape)
      idx[mask] -= dim
    return idx

  return fn

def rand_bool(rng):
  def generator(shape, dtype):
    return _cast_to_shape(
      np.asarray(rng.rand(*_dims_of_shape(shape)) < 0.5, dtype=dtype),
      shape, dtype)
  return generator

def check_raises(thunk, err_type, msg):
  try:
    thunk()
    assert False
  except err_type as e:
    assert str(e).startswith(msg), f"\n{e}\n\n{msg}\n"

def check_raises_regexp(thunk, err_type, pattern):
  try:
    thunk()
    assert False
  except err_type as e:
    assert re.match(pattern, str(e)), f"{e}\n\n{pattern}\n"


