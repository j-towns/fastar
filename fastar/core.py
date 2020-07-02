from typing import Callable, List

import jax.numpy as jnp
import jax.core as jc
from jax.interpreters.xla import abstractify
from jax.util import safe_map, safe_zip
from jax import lax
from fastar.box_util import (
  box_to_slice, slice_to_box, box_finder, static_box_finder, getbox, setbox,
  circular_add, circular_get)
from fastar.jaxpr_util import Literal_, inf
import numpy as np


map = safe_map
zip = safe_zip

UNKNOWN = 0
REQUESTED = -1
KNOWN = 1

CACHE_SIZE = 128

backward_rules = {}

def abstractify_lazy(invals):
  return [abstractify(v.cache) if isinstance(v, LazyArray) else v for v in invals]

class LazyError(Exception): pass

# Circular arrays
class ChildCounts:
  __slots__ = ['buffer', 'total', 'start']
  def __init__(self, shape):
    self.buffer = np.zeros(shape, int)
    self.total = 0
    self.start = np.zeros(len(shape), int)

  def get(self, box):
    box_start, box_shape = box
    box_stop = np.add(box_start, box_shape)
    if self.buffer.shape:
      if (np.all(box_start < np.add(self.start, self.buffer.shape))
          and np.all(box_stop > self.start)):
        pad_lo = np.maximum(np.subtract(self.start, box_start), 0)
        pad_hi = np.maximum(np.add(box_start, box_shape)
                            - np.add(self.start, self.buffer.shape), 0)
        box_stop = np.minimum(np.add(box_start, box_shape),
                              np.add(self.start, self.buffer.shape))
        box_start = np.maximum(self.start, box_start)
        box = box_start, box_stop - box_start
        return np.pad(circular_get(self.buffer, self.start, box),
                      zip(pad_lo, pad_hi))
      else:
        return np.zeros(box_shape, int)
    else:
      assert len(box_start) == len(box_shape) == 0
      return self.buffer

  def add(self, box, val):
    box_start, box_shape = box
    assert val.shape == tuple(box_shape)
    if self.buffer.shape:
      if self.total == 0:
        self.start = box_start
      else:
        self.start = np.minimum(self.start, box_start)
      if np.any(np.add(box_start, box_shape)
                > np.add(self.start, self.buffer.shape)):
        raise LazyError("cache is too small for requested computation.")
      circular_add(self.buffer, self.start, box, val)
    else:
      assert len(box_start) == len(box_shape) == 0
      self.buffer += val
    self.total += np.sum(val)

  def subtract(self, box, val):
    box_start, box_shape = box
    assert val.shape == tuple(box_shape)
    if self.buffer.shape:
      assert np.all(np.add(box_start, box_shape)
                    <= np.add(self.start, self.buffer.shape))
      assert np.all(np.less_equal(self.start, box_start))
      circular_add(self.buffer, self.start, box, -val)
    else:
      assert len(box_start) == len(box_shape) == 0
      self.buffer -= val
    self.total -= np.sum(val)
    assert self.total >= 0


class LazyArray(object):
  __slots__ = ['cache', 'state', 'eqn', 'var_idx', 'child_counts', '_aval']

  def __init__(self, var):
    self._aval = var.aval
    cache_shape = [CACHE_SIZE if d is inf else d for d in self._aval.shape]
    self.cache = jnp.zeros(var.aval.shape, var.aval.dtype)
    self.state = np.zeros(var.aval.shape, int)
    self.child_counts = ChildCounts(cache_shape)
    self.eqn = None
    self.var_idx = None

  def set_eqn(self, eqn):
    assert self.eqn is None and self.var_idx is None
    self.eqn = eqn
    self.var_idx = eqn.outvars.index(self)

  @property
  def shape(self):
    return self._aval.shape

  @property
  def size(self):
    return self._aval.size

  @property
  def dtype(self):
    return self._aval.dtype

  @property
  def ndim(self):
    return self._aval.ndim

  def _compute_ancestral_child_counts(self, box):
    invals, _, primitive, params, _ = self.eqn
    local_state = self.state[box_to_slice(box)] if self.shape else self.state
    to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
    for ubox in box_finder(local_state, UNKNOWN):
      setbox(local_state, ubox, REQUESTED)
      if primitive.multiple_results:
        # TODO: pass var_idx to the backward rule
        raise NotImplementedError
      else:
        instarts, counts, _ = backward_rules[primitive](
            to_global_coords(ubox), *abstractify_lazy(invals), **params)
        for ival, istart, count in zip(invals, instarts, counts):
          if isinstance(ival, LazyArray) and istart is not None:
            ibox = istart, count.shape
            ival.child_counts.add(ibox,
                                  count * (getbox(ival.state, ibox) != KNOWN))
            ival._compute_ancestral_child_counts(ibox)

  def _toposorted_updates(self, box) -> List[Callable[[], None]]:
    self._compute_ancestral_child_counts(box)
    to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
    local_child_counts = self.child_counts.get(box)
    childless_boxes = [
        (self, to_global_coords(b)) for b in static_box_finder(
            (local_child_counts == 0) & (getbox(self.state, box) != KNOWN), 1)]
    sorted_updates = []
    while childless_boxes:
      self._process_childless_box(childless_boxes, sorted_updates)
    return sorted_updates[::-1]

  def _process_childless_box(self, childless_boxes,
                             sorted_updates: List[Callable[[], None]]):
    arr, box = childless_boxes.pop()
    invals, _, primitive, params, _ = arr.eqn
    instarts, counts, outslice_from_inslices = backward_rules[primitive](
      box, *abstractify_lazy(invals), **params)
    def update():
      if primitive.multiple_results:
        raise NotImplementedError
      # TODO (j-towns): add option to disable this assert statement
      # Check that none of this box has already been computed
      assert np.all(getbox(arr.state, box) == REQUESTED), \
        'Repeated computation detected'
      invals_ = [val.cache if isinstance(val, LazyArray) else val
                 for val in invals]
      inslices = (None if instart is None else
                  lax.dynamic_slice(inval, instart, count.shape)
                  for inval, instart, count in zip(invals_, instarts, counts))
      outslice = outslice_from_inslices(*inslices)
      outstart, _ = box
      arr.cache = lax.dynamic_update_slice(arr.cache, outslice, outstart)
      setbox(arr.state, box, KNOWN)
    sorted_updates.append(update)
    for ival, istart, count in zip(invals, instarts, counts):
      if isinstance(ival, LazyArray) and istart is not None:
        ibox = istart, count.shape
        to_iglobal_coords = lambda b: (np.add(istart, b[0]), b[1])
        ival.child_counts.subtract(
          ibox, count * (getbox(ival.state, ibox) != KNOWN))
        ilocal_child_counts = ival.child_counts.get(ibox)
        childless_boxes.extend(
          [(ival, to_iglobal_coords(b))
           for b in static_box_finder((ilocal_child_counts == 0) &
                                      (getbox(ival.state, ibox) != KNOWN), 1)])

  def _getbox(self, box):
    assert np.shape(box) == (2, self.ndim)
    for update in self._toposorted_updates(box):
      update()
    return self.cache[box_to_slice(box)]

  def __getitem__(self, idx):
    if self.size:
      box, int_dims = slice_to_box(self.shape, idx)
      return self._getbox(box)[
          tuple(0 if i in int_dims else slice(None) for i in range(self.ndim))]
    else:
      return jnp.zeros(self.shape, self.dtype)

def lazy_eval_jaxpr(jaxpr, consts, *args):
  def read(v):
    if type(v) in {jc.Literal, Literal_}:
      return v.val
    else:
      return env[v]

  def write(v, val):
    env[v] = val

  env = {}
  write(jc.unitvar, jc.unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      raise NotImplementedError
    map(write, eqn.outvars, map(LazyArray, eqn.outvars))
  for eqn in jaxpr.eqns:
    invals = map(read, eqn.invars)
    outvals = map(read, eqn.outvars)
    new_eqn = jc.JaxprEqn(invals, outvals, eqn.primitive, eqn.params,
                          eqn.source_info)
    map(lambda arr: arr.set_eqn(new_eqn), outvals)
  return map(read, jaxpr.outvars)

def get_aval(x):
  if isinstance(x, LazyArray):
    return x._aval
  else:
    return jc.get_aval(x)
