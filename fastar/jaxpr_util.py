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
  shape: Any
  dtype: Any
  parent: Any
pytype_aval_mappings[DelayedArray] = lambda d: ShapedArray(d.shape, d.dtype)

def abstractify(x):
  return ShapedArray(np.shape(x), dtypes.result_type(x))

def fastar_jaxpr(flat_thunk):
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_thunk, [])
  return ClosedJaxpr(inline_calls(jaxpr), consts)

def tie_the_knot(jaxpr, result):
  assert tuple(jaxpr.out_avals) == (ShapedArray(result.shape_dtype.shape,
                                                result.shape_dtype.dtype),)
  j = jaxpr.jaxpr
  result_var, = j.outvars
  replace_vars = []
  constvars = []
  consts = []
  for v, c in zip(j.constvars, jaxpr.consts):
    if isinstance(c, DelayedArray):
      assert c.parent is result
      replace_vars.append(v)
    else:
      constvars.append(v)
      consts.append(c)
  replace_vars = set(replace_vars)
  replace = lambda v: (
      result_var if (isinstance(v, Var) and v in replace_vars) else v)
  return ClosedJaxpr(
      Jaxpr(constvars, [], j.outvars, [JaxprEqn(
          *([replace(i) for i in e.invars],) + e[1:]) for e in j.eqns]),
      consts)

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
