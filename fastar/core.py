from collections import defaultdict
from typing import Callable, List, Union
import numpy as np
import jax.core as jc
from jax.core import Var
from jax.util import safe_map, safe_zip
from jax import lax, ShapeDtypeStruct, tree_multimap
import jax.numpy as jnp

from fastar.box_util import (box_to_slice, slice_to_box, getbox, setbox,
                             addbox)
from fastar.box_finder import box_finder, static_box_finder


map = safe_map
zip = safe_zip

dependency_rules = {}
meta_argmakers = {}
updaters = {}

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
    # TODO: consider using for loop instead of recursion here
    for start, shape in box_finder(getbox(visited[o], obox), False, True):
      start = np.add(obox[0], start)
      eqn = outvar_to_eqn[o]
      inboxes, counts = dependency_rules[eqn.primitive](
        start, Ones(shape), *map(_shape_dtype, eqn.invars), **eqn.params)
      for i, ibox, count in zip(eqn.invars, inboxes, counts):
        if type(i) is Var and i in child_counts:
          addbox(child_counts[i], ibox, count)
          visit(i, ibox)

  for o in haxpr.outvars:
    visit(o, (o.aval.ndim * (0,), o.aval.shape))
  return child_counts

def toposort(haxpr):
  child_counts = compute_child_counts(haxpr)
  sorted_boxes = {o: [] for e in haxpr.eqns for o in e.outvars}
  childless_boxes = {}
  for o in haxpr.outvars:
    childless_boxes[o] = static_box_finder(child_counts[o], 0)
  while childless_boxes:
    for e in reversed(haxpr.eqns):
      o, = e.outvars
      if o in childless_boxes:
        obox = childless_boxes[o].pop()
        if not childless_boxes[o]:
          childless_boxes.pop(o)
        sorted_boxes[o].append(obox)
        start, shape = obox
        inboxes, counts = dependency_rules[e.primitive](
            start, Ones(shape), *map(_shape_dtype, e.invars), **e.params)
        for i, ibox, count in zip(e.invars, inboxes, counts):
          if isinstance(i, Var) and i in child_counts and ibox is not None:
            assert np.all(count > 0)
            to_iglobal_coords = lambda b: (np.add(ibox[0], b[0]), b[1])
            addbox(child_counts[i], ibox, -materialize(count))
            ilocal_child_counts = getbox(child_counts[i], ibox)
            if not i in childless_boxes:
              childless_boxes[i] = []
            childless_boxes[i].extend(
              [to_iglobal_coords(b)
               for b in static_box_finder((ilocal_child_counts == 0))])
      else:
        sorted_boxes[o].append(None)
    l, = set(map(len, sorted_boxes.values()))  # Check all same length
    sorted_boxes = {o: list(reversed(sorted_boxes[o])) for o in sorted_boxes}
  return sorted_boxes

def sort_and_compress(haxpr):
  sorted_boxes = toposort(haxpr)
  meta_args = []
  for e in haxpr.eqns:
    o, = e.outvars
    meta_argmaker = meta_argmakers[e.primitive]
    static_args, dyn_args = unzip2([
        meta_argmaker(obox, map(_shape_dtype, e.invars), **e.params)
        for obox in sorted_boxes[o]])
    static_args_compressed = list(set(static_args))
    static_args_map = dict(
        (a, i) for i, a in enumerate(static_args_compressed))
    static_args_idxs = np.array(static_args_map[a] for a in static_args)
    dyn_args = tree_multimap(lambda *els: np.concatenate(els), *dyn_args)
    meta_args.append((static_args_compressed, static_args_idxs, dyn_args))
  return meta_args

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

    for (static_args, static_arg_idxs, dyn_args), eqn in zip(meta_args,
                                                             haxpr.eqns):
      invals = map(read, eqn.invars)
      outval, = map(read, eqn.outvars)
      switch_kernels = map(updaters[eqn.primitive], static_args)
      new_outval = lax.switch(static_arg_idxs[i], switch_kernels,
                              (dyn_args[i], invals, eqn.params))
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
