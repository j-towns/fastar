import jax.numpy as jnp
import jax.core as jc
from jax.interpreters import partial_eval as pe
from jax.util import safe_map, safe_zip
from jax import make_jaxpr
from jax import lax
import numpy as np


map = safe_map
zip = safe_zip

def box_to_slice(box):
  return tuple(slice(start, start + size) for start, size in box)

class LazyArray(object):
  __slots__ = ['env', 'cache', 'known', 'primitive', 'invars', 'params']

  def __init__(self, aval, primitive, env, invars, params):
    assert not primitive.multiple_results, "Not implemented"
    self.cache = jnp.array(aval.shape, aval.dtype)
    self.known = np.array(aval.shape, bool)
    self.primitive = primitive
    self.env = env
    self.invars = invars
    self.params = params

  @property
  def shape(self):
    return self.cache.shape

  @property
  def dtype(self):
    return self.cache.dtype

  @property
  def ndim(self):
    return self.cache.ndim

  def getbox(self, box):
    assert np.shape(box) == (self.ndim, 2)
    known = self.known[box_to_slice(box)]
    unkown_boxes = mask_to_boxes(~known)
    for u_box in unknown_boxes:
      inboxes = backward_rules[self.primitive](u_box)
      invals = [read(self.env, invar).getbox(inbox)
                for invar, inbox in zip(invars, inboxes)]
      self.cache = update_rules[self.primitive](u_box, *invals, **params)
      self.known[box_to_slice(u_box)] = True
    return self.cache[box_to_slice(box)]

def tie_the_knot(typed_jaxpr):
  jaxpr, _, in_avals, out_avals = typed_jaxpr
  assert all(i == o for i, o in zip(in_avals, out_avals))
  invars, outvars = jaxpr.invars, jaxpr.outvars
  in_to_out = dict(zip(invars, outvars))
  def replace(eqn):
    invars = [in_to_out[i] if isinstance(i, jc.Var) and i in in_to_out
              else i for i in eqn.invars]
    return jc.JaxprEqn(invars, eqn.outvars, eqn.primitive, eqn.params)
  eqns = [replace(eqn) for eqn in jaxpr.eqns]
  new_jaxpr = jc.Jaxpr(jaxpr.constvars, [], jaxpr.outvars, eqns)
  return jc.TypedJaxpr(new_jaxpr, typed_jaxpr.literals, [],
                       typed_jaxpr.out_avals)

def read(env, v):
  if type(v) is jc.Literal:
    return v.val
  else:
    return env[v]

def lazy_eval_jaxpr(jaxpr, consts, *args):
  def write(v, val):
    env[v] = val

  env = {}
  write(unitvar, unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    in_avals = map(jc.get_aval, eqn.invars)
    call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr:
      raise NotImplementedError
    ans_aval, = primitive.abstract_eval(*in_avals, **eqn.params)
    if eqn.primitive.multiple_results:
      raise NotImplemented
    else:
      write(eqn.outvars[0], LazyArray(ans_aval, eqn.primitive, env, eqn.invars,
                                      eqn.params)
  return map(partial(read, env), jaxpr.outvars)
