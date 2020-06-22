import jax.numpy as jnp
import jax.core as jc
from jax.interpreters import partial_eval as pe
from jax.util import safe_map, safe_zip, partial
from jax import make_jaxpr
from jax import lax
from jax.ops import index_update
from fastar.box_util import (
    box_to_slice, slice_to_box, box_finder, static_box_finder)
import numpy as np


map = safe_map
zip = safe_zip

UNKNOWN = 0
REQUESTED = -1
KNOWN = 1

backward_rules = {}
update_rules = {}

class LazyArray(object):
  __slots__ = ['cache', 'state', 'eqn', 'var_idx', 'child_counts']

  def __init__(self, var):
    self.cache = jnp.zeros(var.aval.shape, var.aval.dtype)
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
    return self.cache.shape

  @property
  def dtype(self):
    return self.cache.dtype

  @property
  def ndim(self):
    return self.cache.ndim

  @property
  def _aval(self):
    return self.cache.aval

  def _compute_ancestral_child_counts(self, box):
    invals, outvals, primitive, params = self.eqn
    in_avals = map(get_aval, invals)
    local_state = self.state[box_to_slice(box)] if self.shape else self.state
    to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
    for u_box in box_finder(local_state, UNKNOWN):
      local_state[box_to_slice(u_box)] = REQUESTED
      if primitive.multiple_results:
        # TODO: pass var_idx to the backward rule
        raise NotImplementedError
      else:
        inboxes, counts = backward_rules[primitive](
            to_global_coords(u_box), *in_avals, **params)
        for ival, ibox, count in zip(invals, inboxes, counts):
          if isinstance(ival, LazyArray) and ibox is not None:
            ival.child_counts[box_to_slice(ibox)] += (
                count * (ival.state[box_to_slice(ibox)] != KNOWN))
            ival._compute_ancestral_child_counts(ibox)

  def _toposort(self, box):
    self._compute_ancestral_child_counts(box)
    to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
    sorted_boxes = []
    local_child_counts = (self.child_counts[box_to_slice(box)] if self.shape
                          else self.child_counts)
    childless_boxes = [
        (self, to_global_coords(b)) for b in static_box_finder(
            (local_child_counts == 0)
            & (self.state[box_to_slice(box)] != KNOWN), 1)]
    while childless_boxes:
      arr, box = childless_boxes.pop()
      sorted_boxes.append((arr, box))
      invals, _, primitive, params = arr.eqn
      in_avals = map(get_aval, invals)
      inboxes, counts = backward_rules[primitive](box, *in_avals, **params)
      for ival, ibox, count in zip(invals, inboxes, counts):
        if isinstance(ival, LazyArray) and ibox is not None:
          to_iglobal_coords = lambda b: (np.add(ibox[0], b[0]), b[1])
          ichild_counts = (ival.child_counts[box_to_slice(ibox)] if ival.shape
                           else ival.child_counts)
          ichild_counts -= (count * (ival.state[box_to_slice(ibox)] != KNOWN))
          childless_boxes.extend(
              [(ival, to_iglobal_coords(b))
               for b in static_box_finder(
                   (ichild_counts == 0) &
                   (ival.state[box_to_slice(ibox)] != KNOWN), 1)])
    return sorted_boxes[::-1]

  def getbox(self, box):
    assert np.shape(box) == (2, self.ndim)
    for arr, u_box in self._toposort(box):
      invals, _, primitive, params = arr.eqn
      if primitive.multiple_results:
        raise NotImplementedError
      else:
        invals = [v.cache if isinstance(v, LazyArray) else v for v in invals]
        arr.cache = update_rules[primitive](arr.cache, u_box, *invals, **params)
        arr.state[box_to_slice(u_box)] = KNOWN
    return self.cache[box_to_slice(box)]

  def __getitem__(self, idx):
    box, int_dims = slice_to_box(self.shape, idx)
    return self.getbox(box)[tuple(0 if i in int_dims else slice(None)
                                  for i in range(self.ndim))]

def get_aval(x):
  if isinstance(x, LazyArray):
    return x._aval
  else:
    return jc.get_aval(x)

def lazy_eval_jaxpr(jaxpr, consts, *args):
  def read(v):
    if type(v) is jc.Literal:
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
    new_eqn = jc.JaxprEqn(invals, outvals, eqn.primitive, eqn.params)
    map(lambda arr: arr.set_eqn(new_eqn), outvals)
  return map(read, jaxpr.outvars)
