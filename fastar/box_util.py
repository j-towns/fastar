"""
Boxes serve as a simplified slice format used internally by FastAR. A box for an
array `a` with `a.ndim == n` is a pair `starts, sizes` where each of starts and
sizes is an iterable with length `n`. The values in `starts` are the start
indices of a slice in each dimension of `a` and the sizes are the sizes of the
slice in each dimension.
"""
import numpy as np
from jax.util import safe_map, safe_zip


map = safe_map
zip = safe_zip

def box_to_slice(box):
  return tuple(slice(start, start + size) for start, size in zip(*box))

def slice_to_box(shape, sl):
  if not isinstance(sl, tuple):
    sl = (sl,) + (len(shape) - 1) * (slice(None),)
  starts, dims = [], []
  int_dims = []
  for i, (arr_dim, s) in enumerate(zip(shape, sl)):
    if not isinstance(s, slice):
      start, stop, step = s, s + 1, None
      int_dims.append(i)
    else:
      start, stop, step = s.start, s.stop, s.step
    if step is not None:
      raise ValueError("Slicing a LazyArray with step != 1 is not supported.")
    start = 0 if start is None else start
    if start < 0 or start > arr_dim:
      raise ValueError("Start of slice is outside of array.")
    stop = arr_dim if stop is None else stop
    if stop < 0 or stop > arr_dim:
      raise ValueError("End of slice is outside of array.")
    starts.append(start)
    dims.append(stop - start)
  return (starts, dims), set(int_dims)

def setbox(arr, box, val):
  arr[box_to_slice(box)] = val

def addbox(arr, box, val):
  arr[box_to_slice(box)] += val

def getbox(arr, box):
  return arr[box_to_slice(box)]

def circular_add(arr, arr_starts, box, val):
  box_starts, box_shape = box
  assert val.shape == tuple(box_shape)
  box_stops = np.add(box_starts, box_shape)
  assert arr.ndim == len(arr_starts) == len(box_starts) == len(box_shape)
  assert np.all(arr_starts <= box_starts)
  assert np.all(box_starts < box_stops)
  assert np.all(box_stops <= np.add(arr_starts, arr.shape))
  starts = np.mod(box_starts, arr.shape)
  stops = np.mod(np.subtract(box_stops, 1), arr.shape) + 1
  arr_starts = np.mod(arr_starts, arr.shape)
  def split_add_(new_slice, val):
    d = len(new_slice)
    if d == arr.ndim:
      arr[tuple(map(lambda s: slice(*s), new_slice))] += val
    else:
      start = starts[d]
      stop = stops[d]
      arr_start = arr_starts[d]
      if stop > arr_start or start < arr_start:
        split_add_(new_slice + [(start, stop)], val)
      else:
        val_lo = val[d * (slice(None),) + (slice(arr.shape[d] - start),)]
        val_hi = val[d * (slice(None),) + (slice(arr.shape[d] - start, None),)]
        split_add_(new_slice + [(start, arr.shape[d])], val_lo)
        split_add_(new_slice + [(0, stop)], val_hi)
  return split_add_([], val)

def circular_get(arr, arr_starts, box):
  # TODO: Speed this up by pre-allocating a box-sized array and copying into it
  box_starts, box_shape = box
  box_stops = np.add(box_starts, box_shape)
  assert arr.ndim == len(arr_starts) == len(box_starts) == len(box_shape)
  assert np.all(arr_starts <= box_starts)
  assert np.all(box_starts < box_stops)
  assert np.all(box_stops <= np.add(arr_starts, arr.shape))
  starts = np.mod(box_starts, arr.shape)
  stops = np.mod(np.subtract(box_stops, 1), arr.shape) + 1
  arr_starts = np.mod(arr_starts, arr.shape)
  def get_(arr, d):
    if d == arr.ndim:
      return arr
    start = starts[d]
    stop = stops[d]
    arr_start = arr_starts[d]
    if stop > arr_start or start < arr_start:
      return get_(arr[d * (slice(None),) + (slice(start, stop),)], d + 1)
    else:
      arr_lo = arr[d * (slice(None),) + (slice(start, None),)]
      arr_hi = arr[d * (slice(None),) + (slice(stop),)]
      return np.concatenate([get_(arr_lo, d + 1), get_(arr_hi, d + 1)], d)
  return get_(arr, 0)

def test_boxes(starts, sizes, dim):
  assert sizes[dim] == 1
  i = 1
  while True:
    yield tuple(start + i if d == dim else slice(start, start + size)
                for d, (start, size) in enumerate(zip(starts, sizes)))
    i = i + 1

def box_finder(known, value):
  it = np.nditer(known, flags=['multi_index'])
  for k in it:
    if k == value:
      starts = it.multi_index
      sizes = known.ndim * [1]
      for d in range(known.ndim):
        box_iter = test_boxes(starts, sizes, d)
        while (starts[d] + sizes[d] < known.shape[d] and
               np.all(known[next(box_iter)] == value)):
          sizes[d] = sizes[d] + 1
      yield starts, sizes

def static_box_finder(known, value):
  tmp = known == value
  ret = []
  for box in box_finder(tmp, 1):
    ret.append(box)
    if tmp.shape:
      tmp[box_to_slice(box)] = 0
  return list(ret)
