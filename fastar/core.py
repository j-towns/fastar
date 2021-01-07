from functools import partial, cached_property
from dataclasses import dataclass

from collections import defaultdict
from typing import Callable, List, Union, Any
import numpy as np
import jax.core as jc
from jax.core import Var
from jax.util import safe_map, safe_zip, unzip2
from jax import lax, ShapeDtypeStruct, tree_multimap, tree_map
import jax.numpy as jnp

from fastar.box_util import box_to_slice, slice_to_box, getbox, setbox, addbox
from fastar.box_finder import box_finder, static_box_finder


map = safe_map
zip = safe_zip

dependency_rules = {}
kernels = {}

@dataclass(frozen=True)
class UpdateMeta:
  # TODO: Type these
  outbox: Any
  inboxes: Any
  meta_static: Any
  meta_dynamic: Any

  @cached_property
  def outstart(self):
    outstart, _ = self.outbox
    return jnp.array(outstart)

  @cached_property
  def outshape(self):
    _, outshape = self.outbox
    return tuple(outshape)

  @cached_property
  def instarts(self):
    return [null if s is None else jnp.array(s[0]) for s in self.inboxes]

  @cached_property
  def inshapes(self):
    return tuple(s and tuple(s[1]) for s in self.inboxes)

def _shape_dtype(v):
  return ShapeDtypeStruct(v.aval.shape, v.aval.dtype)

def compute_child_counts(haxpr):
  outvar_to_eqn = {}
  visited = {}
  child_counts = {}
  for eqn in haxpr.eqns:
    o, = eqn.outvars
    outvar_to_eqn[o] = eqn
    visited[o]       = np.zeros(o.aval.shape, bool)
    child_counts[o]  = np.zeros(o.aval.shape, int)

  def visit(o, obox):
    # TODO: consider using a for loop instead of recursion here
    for start, shape in box_finder(getbox(visited[o], obox), False, True):
      start = np.add(obox[0], start)
      eqn = outvar_to_eqn[o]
      inboxes, counts, _ = dependency_rules[eqn.primitive](
        start, Ones(shape), *map(_shape_dtype, eqn.invars), **eqn.params)
      for i, ibox, count in zip(eqn.invars, inboxes, counts):
        if type(i) is Var and i in child_counts:
          addbox(child_counts[i], ibox, materialize(count))
          visit(i, ibox)

  for o in haxpr.outvars:
    visit(o, (o.aval.ndim * (0,), o.aval.shape))
  return child_counts

def toposort(haxpr):
  child_counts = compute_child_counts(haxpr)
  sorted_updates = [[] for _ in haxpr.eqns]
  childless_boxes = {}
  for o in haxpr.outvars:
    childless_boxes[o] = static_box_finder(child_counts[o], 0)
  while childless_boxes:
    for e_sorted, e in reversed(zip(sorted_updates, haxpr.eqns)):
      o, = e.outvars
      if o in childless_boxes:
        obox = childless_boxes[o].pop()
        if not childless_boxes[o]:
          childless_boxes.pop(o)
        start, shape = obox
        inboxes, counts, (m_static, m_dyn) = dependency_rules[e.primitive](
            start, Ones(shape), *map(_shape_dtype, e.invars), **e.params)
        e_sorted.append(UpdateMeta(obox, inboxes, m_static, m_dyn))
        for i, ibox, count in zip(e.invars, inboxes, counts):
          if isinstance(i, Var) and i in child_counts and ibox is not None:
            count = materialize(count)
            assert np.all(count > 0)
            to_iglobal_coords = lambda b: (np.add(ibox[0], b[0]), b[1])
            addbox(child_counts[i], ibox, -count)
            ilocal_child_counts = getbox(child_counts[i], ibox)
            if not i in childless_boxes:
              childless_boxes[i] = []
            childless_boxes[i].extend(
              [to_iglobal_coords(b)
               for b in static_box_finder((ilocal_child_counts == 0))])
      else:
        e_sorted.append(None)
  assert len(set(map(len, sorted_updates))) == 1
  assert all(np.all(v == 0) for v in child_counts.values())
  return [list(reversed(s)) for s in sorted_updates]

def _any(els):
  for el in els:
    if el: return el
  return el

# We use this because None is treated as a pytree node and we need a leaf
class NullType: pass
null = NullType()

def _stack(*args):
  sup = null
  for a in args:
    if a is not null:
      sup = a
  return (None if sup is null
          else jnp.stack([sup if a is null else a for a in args]))

def compress(updates):
  static = [u and (u.meta_static, u.inshapes) for u in updates]
  static_compressed = list(set(static))
  static_mapping = dict((a, i) for i, a in enumerate(static_compressed))
  static_idxs = jnp.array([static_mapping[a] for a in static])
  dynamic = [u and (u.meta_dynamic, u.instarts, u.outstart) for u in updates]
  sup = _any(dynamic)
  dynamic = sup and tree_multimap(_stack, *[d or sup for d in dynamic])
  return static_compressed, static_idxs, dynamic

def sort_and_compress(haxpr):
  return map(compress, toposort(haxpr))

def _identity_update(switch_operand):
  _, old_outval, *_ = switch_operand
  return old_outval

def make_updater(kernel, static):
  if static is None: return _identity_update
  meta_static, inshapes = static
  def update(switch_operand):
    dynamic, old_outval, *invals = switch_operand
    meta_dynamic, instarts, outstart = dynamic
    invals = [None if shape is None else lax.dynamic_slice(i, start, shape)
              for i, start, shape in zip(invals, instarts, inshapes)]
    return lax.dynamic_update_slice(
         old_outval, kernel(meta_static, meta_dynamic, *invals), outstart)
  return update

def eval_haxpr(haxpr, consts):
  meta_args = sort_and_compress(haxpr)
  def write_const(v, val):
    assert v not in const_env
    const_env[v] = val

  def write(v, val):
    assert v not in env
    env[v] = val

  def read_final(v):
    if type(v) is jc.Literal:
      return v.val
    elif v in const_env:
      return const_env[v]
    else:
      return final_env[v]

  env = {}
  const_env = {}
  write_const(jc.unitvar, jc.unit)
  map(write_const, haxpr.constvars, consts)
  for eqn in haxpr.eqns:
    map(write, eqn.outvars,
        [jnp.zeros(o.aval.shape, o.aval.dtype) for o in eqn.outvars])

  def body_fun(i, env):
    env = env.copy()
    def read(v):
      if type(v) is jc.Literal:
        return v.val
      elif v in const_env:
        return const_env[v]
      else:
        return env[v]

    def overwrite(v, val):
      assert v in env
      env[v] = val

    for (static, static_idxs, dynamic), eqn in zip(meta_args, haxpr.eqns):
      invals = map(read, eqn.invars)
      outval, = map(read, eqn.outvars)
      updaters = map(partial(make_updater, kernels[eqn.primitive]), static)
      new_outval = lax.switch(
          static_idxs[i], updaters,
          (tree_map(lambda x: None if x is None else x[i], dynamic), outval,
           *invals))
      overwrite(eqn.outvars[0], new_outval)
    return env
  num_iters = len(meta_args[0][1] if meta_args else 0)
  final_env = lax.fori_loop(0, num_iters, body_fun, env)
  return map(read_final, haxpr.outvars)

class Ones:
  def __init__(self, shape):
    self.shape = shape

  ndim = property(lambda self: len(self.shape))
  size = property(lambda self: np.prod(self.shape, dtype=int))

  def __eq__(self, other):
    return is_ones(other) and self.shape == other.shape

def is_ones(count):
  return type(count) is Ones

def materialize(count):
  return np.ones(count.shape, int) if is_ones(count) else count
