from dataclasses import dataclass
from typing import Any

import numpy as np

from jax import (ShapedArray, abstract_arrays, dtypes, core as jc,
                 linear_util as lu)
from jax.util import safe_map, safe_zip
from jax.core import (Literal, Jaxpr, JaxprEqn, Var, ClosedJaxpr,
                      pytype_aval_mappings)
from jax.interpreters import xla, partial_eval as pe
from jax.abstract_arrays import ShapedArray
from functools import partial

map = safe_map
zip = safe_zip

@dataclass
class DelayedArray:
  # TODO: fix infix operators (will need operator overloading here)
  shape: Any
  dtype: Any
  parent: Any
  idx: Any
pytype_aval_mappings[DelayedArray] = lambda d: ShapedArray(d.shape, d.dtype)
xla.pytype_aval_mappings[DelayedArray] = lambda d: ShapedArray(d.shape, d.dtype)

def abstractify(x):
  return ShapedArray(np.shape(x), dtypes.result_type(x))

def fastar_jaxpr(flat_thunk):
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_thunk, [])
  return ClosedJaxpr(inline_calls(jaxpr), consts)

def tie_the_knot(closed_jaxpr, result):
  jaxpr = closed_jaxpr.jaxpr
  to_replace = {}
  new_constvars = []
  new_consts = []
  for v, c in zip(jaxpr.constvars, closed_jaxpr.consts):
    if isinstance(c, DelayedArray):
      assert c.parent is result
      to_replace[v] = jaxpr.outvars[c.idx]
    else:
      new_constvars.append(v)
      new_consts.append(c)
  replace = lambda v: (
      to_replace[v] if (isinstance(v, Var) and v in to_replace) else v)
  return ClosedJaxpr(
      Jaxpr(new_constvars, [], jaxpr.outvars,
            [JaxprEqn(*([replace(i) for i in e.invars],) + e[1:])
             for e in jaxpr.eqns]),
      new_consts)

def inline_calls(jaxpr):
  new_eqns = []

  def inline_call(jaxpr, invars, outvars):
    inmap = dict(zip(jaxpr.invars, invars))
    outmap = dict(zip(jaxpr.outvars, outvars))
    for eqn in jaxpr.eqns:
      new_invars = [v if isinstance(v, Literal) else inmap.get(v, v)
                    for v in eqn.invars]
      new_outvars = [outmap.get(v, v) for v in eqn.outvars]
      call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
      if call_jaxpr:
        if not eqn.primitive in {jc.call_p, xla.xla_call_p}:
          raise NotImplementedError
        inline_call(call_jaxpr, new_invars, new_outvars)
      else:
        new_eqns.append(
            JaxprEqn(new_invars, new_outvars, eqn.primitive, eqn.params,
                     eqn.source_info))

  for eqn in jaxpr.eqns:
    call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      if not eqn.primitive in {jc.call_p, xla.xla_call_p}:
        raise NotImplementedError
      inline_call(call_jaxpr, eqn.invars, eqn.outvars)
    else:
      new_eqns.append(eqn)

  return refresh_names(
      Jaxpr(jaxpr.constvars, jaxpr.invars, jaxpr.outvars, new_eqns))

def refresh_names(jaxpr):
  vs = {}
  g = jc.gensym()
  varmap = lambda v: vs[v] if v in vs else vs.setdefault(v, g(v.aval))
  jaxpr_constvars = map(varmap, jaxpr.constvars)
  jaxpr_invars = map(varmap, jaxpr.invars)
  new_eqns = []
  for eqn in jaxpr.eqns:
    invars = [v if isinstance(v, Literal) else varmap(v) for v in eqn.invars]
    outvars = map(varmap, eqn.outvars)
    new_eqns.append(
        JaxprEqn(invars, outvars, eqn.primitive, eqn.params, eqn.source_info))
  jaxpr_outvars = map(varmap, jaxpr.outvars)
  return Jaxpr(jaxpr_constvars, jaxpr_invars, jaxpr_outvars, new_eqns)
