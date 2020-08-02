from typing import Callable, List, Union
import numpy as np
import jax.core as jc
from jax.lax.lax import Array
from jax.util import safe_map, safe_zip
from jax import lax, numpy as jnp
from jax.interpreters.numpy_eval import numpy_eval

from fastar.box_util import (box_to_slice, slice_to_box, getbox, setbox,
                             addbox)
from fastar.box_finder import box_finder, static_box_finder
from fastar.jaxpr_util import Literal_
from fastar.numpy_eval_util import inplace_dynamic_update_slice

map = safe_map
zip = safe_zip

UNKNOWN = 0
REQUESTED = -1
KNOWN = 1

dependency_rules = {}

class LazyArray(object):
  __slots__ = ['cache', 'state', 'eqn', 'var_idx', 'child_counts', '_aval',
               'todo']

  def __init__(self, var):
    self._aval = var.aval
    self.cache = np.zeros(var.aval.shape, var.aval.dtype)
    self.state = np.zeros(var.aval.shape, int)
    self.child_counts = np.zeros(var.aval.shape, int)
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
    start, shape = box
    invals, _, primitive, params, _ = self.eqn
    local_state = getbox(self.state, box)
    uboxes = box_finder(local_state, UNKNOWN, REQUESTED)
    for ubox in uboxes:
      if primitive.multiple_results:
        # TODO: pass var_idx to the dependency rule
        raise NotImplementedError
      ustart, ushape = ubox
      ustart = np.add(start, ustart)
      instarts, counts, _ = dependency_rules[primitive](
        ustart, Ones(ushape), *invals, **params)
      for ival, istart, count in zip(invals, instarts, counts):
        if isinstance(ival, LazyArray) and istart is not None:
          ibox = istart, np.shape(count)
          addbox(ival.child_counts,
            ibox, materialize(count) * (getbox(ival.state, ibox) != KNOWN))
          ival._compute_ancestral_child_counts(ibox)

  def _toposorted_updates(self, box) -> List[Callable[[], None]]:
    self._compute_ancestral_child_counts(box)
    to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
    local_child_counts = getbox(self.child_counts, box)
    childless_boxes = [
        (self, to_global_coords(b)) for b in static_box_finder(
            (local_child_counts == 0) & (getbox(self.state, box) != KNOWN))]
    sorted_updates = []
    while childless_boxes:
      arr, box = childless_boxes.pop()
      update, instarts, counts = make_update_thunk(arr, box)
      sorted_updates.append(update)
      for ival, istart, count in zip(arr.eqn.invars, instarts, counts):
        if isinstance(ival, LazyArray) and istart is not None:
          ibox = istart, np.shape(count)
          to_iglobal_coords = lambda b: (np.add(istart, b[0]), b[1])
          addbox(ival.child_counts,
            ibox, -materialize(count) * (getbox(ival.state, ibox) != KNOWN))
          ilocal_child_counts = getbox(ival.child_counts, ibox)
          childless_boxes.extend(
            [(ival, to_iglobal_coords(b))
             for b in static_box_finder((ilocal_child_counts == 0) &
                                        (getbox(ival.state, ibox) != KNOWN))])
    return sorted_updates[::-1]

  def _getbox(self, box):
    assert np.shape(box) == (2, self.ndim)
    for update in self._toposorted_updates(box):
      update()
    return self.cache[box_to_slice(box)]

  @numpy_eval()
  def __getitem__(self, idx):
    if self.size:
      box, int_dims = slice_to_box(self.shape, idx)
      return self._getbox(box)[
          tuple(0 if i in int_dims else slice(None) for i in range(self.ndim))]
    else:
      return jnp.zeros(self.shape, self.dtype)

def make_update_thunk(arr, box):
  start, shape = box
  invals, _, primitive, params, _ = arr.eqn
  instarts, counts, outslice_from_inslices = dependency_rules[primitive](
    start, Ones(shape), *invals, **params)
  def thunk():
    if primitive.multiple_results:
      raise NotImplementedError
    # TODO (j-towns): add option to disable this assert statement
    assert np.all(getbox(arr.state, box) == REQUESTED), \
      'Repeated computation detected'
    invals_ = [val.cache if isinstance(val, LazyArray) else jnp.asarray(val)
               for val in invals]
    inslices = [None if instart is None else
                lax.slice(inval, instart, np.array(instart) + np.shape(count))
                for inval, instart, count in zip(invals_, instarts, counts)]
    outslice = outslice_from_inslices(*inslices)
    outstart, _ = box
    arr.cache = inplace_dynamic_update_slice(arr.cache, outslice, outstart)
    setbox(arr.state, box, KNOWN)
  return thunk, instarts, counts

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
  return x._aval if isinstance(x, LazyArray) else jc.get_aval(x)

class Ones:
  def __init__(self, shape):
    self.shape = shape

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: np.prod(self.shape, dtype=int))

  def __eq__(self, other):
    return is_ones(other) and self.shape == other.shape

def is_ones(count: Union[Array, Ones]):
  return type(count) is Ones

def materialize(count: Union[Array, Ones]):
  return np.ones(count.shape, int) if is_ones(count) else count
