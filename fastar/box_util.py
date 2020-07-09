"""
Boxes serve as a simplified slice format used internally by FastAR. A box for an
array `a` with `a.ndim == n` is a pair `starts, sizes` where each of starts and
sizes is an iterable with length `n`. The values in `starts` are the start
indices of a slice in each dimension of `a` and the sizes are the sizes of the
slice in each dimension.
"""
import numpy as np
from jax.util import safe_map, safe_zip

from itertools import product

_map = map

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

def ndindex(starts, stops):
  return product(*_map(range, starts, stops))

def _contains_box(idxs, starts, stops):
  return idxs.issuperset(ndindex(starts, stops))

def _remove_box(idxs, starts, stops):
  idxs.difference_update(ndindex(starts, stops))

def box_finder(idxs):
  while idxs:
    starts = min(idxs, key=lambda i: tuple(reversed(i)))
    idxs.remove(starts)
    starts = list(starts)
    stops = [s + 1 for s in starts]
    for d in range(len(starts)):
      test_starts = starts.copy()
      test_stops = stops.copy()
      test_starts[d] += 1
      test_stops[d] += 1
      while _contains_box(idxs, test_starts, test_stops):
        stops[d] += 1
        _remove_box(idxs, test_starts, test_stops)
        test_starts[d] += 1
        test_stops[d] += 1
    yield starts, [stop - start for start, stop in zip(starts, stops)]

def static_box_finder(arr, val=0):
  return list(box_finder(set(map(tuple, np.argwhere(arr == val)))))
