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
    tmp[box_to_slice(box)] = 0
  return list(ret)
