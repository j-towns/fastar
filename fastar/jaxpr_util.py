import numpy as np

from jax import linear_util as lu
from jax.tree_util import tree_unflatten, tree_flatten
from jax.util import safe_map, unzip2
from jax.core import Literal, Jaxpr, JaxprEqn, Var, TypedJaxpr
from jax import tree_util
from jax.interpreters import xla
import jax.interpreters.partial_eval as pe
from jax import ShapedArray
from jax import abstract_arrays
import jax.core as jc
from jax import dtypes
from functools import partial

import numpy as onp
from jax import tree_util


map = safe_map

class InfShapeError(Exception): pass

class InfType:
  neg: bool

  def __init__(self, neg=False):
    self.neg = neg

  def __add__(self, other):
    if isinstance(other, InfType) and self.neg != other.neg:
      raise InfShapeError
    else:
      return _neginf if self.neg else inf

  def __sub__(self, other):
    if isinstance(other, InfType) and self.neg == other.neg:
      raise InfShapeError
    else:
      return _neginf if self.neg else inf

  def __neg__(self):
    if self.neg:
      return inf
    else:
      return _neginf

  def __mul__(self, other):
    if not isinstance(other, InfType) and other == 0:
      raise InfShapeError
    other_neg = other.neg if isinstance(other, InfType) else other < 0
    return inf if other_neg == self.neg else _neginf

  def __str__(self):
    if self.neg:
      return '-inf'
    else:
      return 'inf'

  def __repr__(self):
    return __str__(self)


abstract_arrays._DIMENSION_TYPES.add(InfType)

inf = InfType()
_neginf = InfType(neg=True)

def fastar_jaxpr(flat_fun, *args_flat):
  def abstractify(x):
    return ShapedArray(np.shape(x), dtypes.result_type(x))
  in_avals = map(abstractify, args_flat)
  in_pvals = map(pe.PartialVal.unknown, in_avals)
  jaxpr, out_pvals, consts = pe.trace_to_jaxpr(
      flat_fun, in_pvals, instantiate=True, stage_out=True)
  out_avals = [v.get_aval() for v in out_pvals]
  return TypedJaxpr(submerge_consts(jaxpr, consts), [], in_avals, out_avals)

def tie_the_knot(typed_jaxpr):
  jaxpr, _, in_avals, out_avals = typed_jaxpr
  assert all(i == o for i, o in zip(in_avals, out_avals))
  in2out = dict(zip(jaxpr.invars, jaxpr.outvars))
  def replace(eqn):
    invars = [in2out[i] if (isinstance(i, jc.Var) and i in in2out) else i
              for i in eqn.invars]
    return jc.JaxprEqn(invars, eqn.outvars, eqn.primitive, eqn.params,
                       eqn.source_info)
  eqns = [replace(eqn) for eqn in jaxpr.eqns]
  new_jaxpr = jc.Jaxpr(jaxpr.constvars, [], jaxpr.outvars, eqns)
  return jc.TypedJaxpr(new_jaxpr, typed_jaxpr.literals, [],
                       typed_jaxpr.out_avals)

# Move constants inside jaxpr, i.e. make them into 'literals'
# Need a custom literal class because jax.core.literal only supports scalars
class Literal_(Literal):
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  @property
  def aval(self):
    return raise_to_shaped(get_aval(self.val))

  def __hash__(self):
    return id(self.val)

  def __eq__(self, other):
    return self.val is other.val

  def __repr__(self):
    return '{}'.format(self.val)


def submerge_consts(jaxpr, consts, invals=None):
  """
  Replace constvars with literals in jaxpr and its sub-jaxprs.
  """
  # TODO(j-towns): check that consts are in jax.core.literalable_types
  consts = dict(zip(jaxpr.constvars, consts))
  if invals is not None:
    # We're in a call_jaxpr
    new_jaxpr_invars = []
    for var, val in zip(jaxpr.invars, invals):
      if isinstance(val, Var):
        new_jaxpr_invars.append(var)
      else:
        consts[var] = val
  else:
    new_jaxpr_invars = jaxpr.invars
  new_eqns = []
  for eqn in jaxpr.eqns:
    if all(isinstance(var, Literal) or var in consts for var in eqn.invars):
      # Perform constant folding if all inputs to an eqn are known
      in_vals = [var.val if isinstance(var, Literal) else consts[var]
                 for var in eqn.invars]
      call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
      if call_jaxpr:
        subfuns = [lu.wrap_init(partial(jc.eval_jaxpr, call_jaxpr, ()))]
      else:
        subfuns = []
      ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
      if eqn.primitive.multiple_results:
        for outvar, out in zip(eqn.outvars, ans):
          consts[outvar] = out
      else:
        outvar, = eqn.outvars
        consts[outvar] = ans
    else:
      new_invars = [consts[var] if (isinstance(var, Var) and var in consts)
                    else var for var in eqn.invars]
      new_params = dict(eqn.params)
      if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
        new_params['call_jaxpr'] = submerge_consts(eqn.params['call_jaxpr'], [],
                                                   new_invars)
        new_invars = [var for var in new_invars if isinstance(var, Var)]
      else:
        new_invars = [var if isinstance(var, (Var, Literal)) else Literal_(var)
                      for var in new_invars]
      new_eqns.append(JaxprEqn(invars=new_invars, outvars=eqn.outvars,
                               primitive=eqn.primitive, params=new_params,
                               source_info=eqn.source_info))
  return Jaxpr([], new_jaxpr_invars, jaxpr.outvars, new_eqns)
