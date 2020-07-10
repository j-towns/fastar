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

def update_trie(trie, idxs):
  for idx in idxs:
    branch = trie
    for i in idx:
      branch = branch[i] if i in branch else {}

def _contains_rectangle(idx_trie, rectangle):
  """
  Return True if rectangle is contained in idx_trie, else False.
  """
  (start, stop), rectangle = rectangle
  return all(
    n in idx_trie
    and (not rectangle or _contains_rectangle(idx_trie[n], rectangle))
    for n in range(start, stop))

def _remove_rectangle(idx_trie, rectangle):
  (start, stop), rectangle = rectangle
  for root in list(idx_trie):
    if start <= root < stop:
      if rectangle:
        _remove_rectangle(idx_trie[root], rectangle)
      if not idx_trie[root]:
        del idx_trie[root]

def _find_rectangle(idx_trie):
  """
  Greedily find a rectangle in idx_trie.
  """
  start = min(idx_trie)
  stop = start + 1
  branch = idx_trie[start]
  if branch:
    rect = _find_rectangle(branch)
    while stop in idx_trie and _contains_rectangle(idx_trie[stop], rect):
      stop += 1
    return (start, stop), rect
  else:
    while stop in idx_trie:
      stop += 1
    return (start, stop), ()

def _to_box(rectangle):
  starts = []
  shape = []
  while rectangle:
    (start, stop), rectangle = rectangle
    starts.append(start)
    shape.append(stop - start)
  return starts, shape

def static_box_finder(arr, val=0):
  """
  Greedily search for boxes in arr.
  """
  if np.shape(arr) == ():
    return [([], [])] if arr else []

  rectangles = []
  idx_trie = {}
  update_trie(idx_trie, np.argwhere(arr == val))
  while idx_trie:
    rect = _find_rectangle(idx_trie)
    rectangles.append(rect)
    _remove_rectangle(idx_trie, rect)
  return map(_to_box, rectangles)

def box_finder(idx_trie):
  while idx_trie:
    rect = _find_rectangle(idx_trie)
    _remove_rectangle(idx_trie, rect)
    yield _to_box(rect)
