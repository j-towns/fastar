import numpy as np

from jax import (ShapedArray, abstract_arrays, dtypes, core as jc,
                 linear_util as lu)
from jax.util import safe_map, safe_zip
from jax.core import Literal, Jaxpr, JaxprEqn, Var, ClosedJaxpr
from jax.interpreters import xla, partial_eval as pe
from functools import partial

map = safe_map
zip = safe_zip

def abstractify(x):
  return ShapedArray(np.shape(x), dtypes.result_type(x))

def fastar_jaxpr(flat_fun, *args_flat):
  in_avals = map(abstractify, args_flat)
  jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
  return ClosedJaxpr(inline_calls(jaxpr), consts)

def tie_the_knot(jaxpr):
  """
  Assuming jaxpr.in_avals == jaxpr.out_avals, replace references to any of
  jaxpr's invars with the corresponding outvar.
  """
  assert tuple(jaxpr.in_avals) == tuple(jaxpr.out_avals)
  j = jaxpr.jaxpr
  in2out = dict(zip(j.invars, j.outvars))
  replace = lambda v: in2out.get(v, v) if isinstance(v, Var) else v
  return ClosedJaxpr(
      Jaxpr(j.constvars, [], j.outvars, [JaxprEqn(
          *([replace(i) for i in e.invars],) + e[1:]) for e in j.eqns]),
      jaxpr.consts)

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
