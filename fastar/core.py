from collections import defaultdict
from typing import Callable, List, Union
import numpy as np
import jax.core as jc
from jax.core import Var
from jax.lax.lax import Array
from jax.util import safe_map, safe_zip
from jax import lax, ShapeDtypeStruct
import jax.numpy as jnp

from fastar.box_util import (box_to_slice, slice_to_box, getbox, setbox,
                             addbox)
from fastar.box_finder import box_finder, static_box_finder


map = safe_map
zip = safe_zip

dependency_rules = {}

def _shape_dtype(v):
  return ShapeDtypeStruct(v.aval.shape, v.aval.dtype)

def compute_child_counts(jaxpr):
  outvar_to_eqn = {}
  visited = {}
  child_counts = {}
  for eqn in jaxpr.eqns:
    for o in eqn.outvars:
      outvar_to_eqn[o] = eqn
      visited[o] = np.zeros(o.aval.shape, bool)
      child_counts[o] = np.zeros(o.aval.shape, int)

  def visit(o, obox):
    for start, shape in box_finder(getbox(visited[o], obox), False, True):
      start = np.add(obox[0], start)
      eqn = outvar_to_eqn[o]
      primitive = eqn.primitive
      if primitive.multiple_results:
        # TODO: pass outvar index to the dependency rule
        raise NotImplementedError
      inboxes, counts, _ = dependency_rules[primitive](
        start, Ones(shape), *map(_shape_dtype, eqn.invars), **eqn.params)
      for i, ibox, count in zip(eqn.invars, inboxes, counts):
        if type(i) is Var and i in child_counts:
          addbox(child_counts[i], ibox, count)
          visit(i, ibox)

  for o in jaxpr.outvars:
    visit(o, (o.aval.ndim * (0,), o.aval.shape))
  return child_counts

def toposort(jaxpr):
  child_counts = compute_child_counts(jaxpr)
  sorted_boxes = {o: [] for e in jaxpr.eqns for o in e.outvars}
  childless_boxes = defaultdict(list)
  for o in jaxpr.outvars:
    childless_boxes[o] = static_box_finder(child_counts[o], 0)
  outvars = [(e, o) for e in reversed(jaxpr.eqns) for o in e.outvars]
  # from IPython.terminal.debugger import set_trace; set_trace()
  while childless_boxes:
    for e, o in outvars:
      # from IPython.terminal.debugger import set_trace; set_trace()
      childless = childless_boxes[o]
      del childless_boxes[o]
      sorted_boxes[o].append(childless)
      for obox in childless:
        primitive = e.primitive
        if primitive.multiple_results:
          # TODO: pass outvar index to the dependency rule
          raise NotImplementedError
        start, shape = obox
        inboxes, counts, _ = dependency_rules[primitive](
            start, Ones(shape), *map(_shape_dtype, e.invars), **e.params)
        for i, ibox, count in zip(e.invars, inboxes, counts):
          if isinstance(i, Var) and i in child_counts and ibox is not None:
            assert np.all(count > 0)
            to_iglobal_coords = lambda b: (np.add(ibox[0], b[0]), b[1])
            addbox(child_counts[i], ibox, -materialize(count))
            ilocal_child_counts = getbox(child_counts[i], ibox)
            childless_boxes[i].extend(
              [to_iglobal_coords(b)
               for b in static_box_finder((ilocal_child_counts == 0))])
  return sorted_boxes

#   def _toposorted_updates(self, box) -> List[Callable[[], None]]:
#     self._compute_ancestral_child_counts(box)
#     to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
#     local_child_counts = getbox(self.child_counts, box)
#     childless_boxes = [
#         (self, to_global_coords(b)) for b in static_box_finder(
#             (local_child_counts == 0) & (getbox(self.state, box) != KNOWN))]
#     sorted_updates = []
#     while childless_boxes:
#       arr, box = childless_boxes.pop()
#       update, inboxes, counts = make_update_thunk(arr, box)
#       sorted_updates.append(update)
#       for ival, ibox, count in zip(arr.eqn.invars, inboxes, counts):
#         if isinstance(ival, LazyArray) and ibox is not None:
#           to_iglobal_coords = lambda b: (np.add(ibox[0], b[0]), b[1])
#           addbox(ival.child_counts,
#             ibox, -materialize(count) * (getbox(ival.state, ibox) != KNOWN))
#           ilocal_child_counts = getbox(ival.child_counts, ibox)
#           childless_boxes.extend(
#             [(ival, to_iglobal_coords(b))
#              for b in static_box_finder((ilocal_child_counts == 0) &
#                                         (getbox(ival.state, ibox) != KNOWN))])
#     return sorted_updates[::-1]
# 
#   def _getbox(self, box):
#     assert np.shape(box) == (2, self.ndim)
#     for update in self._toposorted_updates(box):
#       update()
#     return self.cache[box_to_slice(box)]
# 
#   def __getitem__(self, idx):
#     if self.size:
#       box, int_dims = slice_to_box(self.shape, idx)
#       return self._getbox(box)[
#           tuple(0 if i in int_dims else slice(None) for i in range(self.ndim))]
#     else:
#       return jnp.zeros(self.shape, self.dtype)
# 
# def make_update_thunk(arr, box):
#   start, shape = box
#   invals, _, primitive, params, _ = arr.eqn
#   inboxes, counts, outslice_from_inslices = dependency_rules[primitive](
#     start, Ones(shape), *invals, **params)
#   def thunk():
#     if primitive.multiple_results:
#       raise NotImplementedError
#     # TODO (j-towns): add option to disable this assert statement
#     assert np.all(getbox(arr.state, box) == REQUESTED), \
#       'Repeated computation detected'
#     invals_ = [val.cache if isinstance(val, LazyArray) else jnp.asarray(val)
#                for val in invals]
#     inslices = [None if ibox is None else
#                 lax.slice(inval, ibox[0], np.add(ibox[0], ibox[1]))
#                 for inval, ibox, count in zip(invals_, inboxes, counts)]
#     outslice = outslice_from_inslices(*inslices)
#     outstart, _ = box
#     arr.cache = inplace_dynamic_update_slice(arr.cache, outslice, outstart)
#     setbox(arr.state, box, KNOWN)
#   return thunk, inboxes, counts
# 
# def lazy_eval_jaxpr(jaxpr, consts, *args):
#   def read(v):
#     if type(v) is jc.Literal:
#       return v.val
#     else:
#       return env[v]
# 
#   def write(v, val):
#     env[v] = val
# 
#   env = {}
#   write(jc.unitvar, jc.unit)
#   map(write, jaxpr.constvars, consts)
#   map(write, jaxpr.invars, args)
#   for eqn in jaxpr.eqns:
#     call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
#     if call_jaxpr:
#       raise NotImplementedError
#     map(write, eqn.outvars, map(LazyArray, eqn.outvars))
#   for eqn in jaxpr.eqns:
#     invals = map(read, eqn.invars)
#     outvals = map(read, eqn.outvars)
#     new_eqn = jc.JaxprEqn(invals, outvals, eqn.primitive, eqn.params,
#                           eqn.source_info)
#     map(lambda arr: arr.set_eqn(new_eqn), outvals)
#   return map(read, jaxpr.outvars)
# 
# def get_aval(x):
#   return x._aval if isinstance(x, LazyArray) else jc.get_aval(x)
# 
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
