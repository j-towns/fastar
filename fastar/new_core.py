import jax.numpy as jnp
import jax.core as jc
from jax.interpreters import partial_eval as pe
from jax.util import safe_map, safe_zip, partial
from jax import make_jaxpr
from jax import lax
from jax.ops import index_update
import numpy as np


map = safe_map
zip = safe_zip

UNKNOWN = 0
REQUESTED = -1
KNOWN = 1

def box_to_slice(box):
  return tuple(slice(start, start + size) for start, size in zip(*box))

backward_rules = {}
update_rules = {}

class LazyArray(object):
  __slots__ = ['cache', 'state', 'eqn', 'var_idx']

  def __init__(self, var):
    self.cache = jnp.zeros(var.aval.shape, var.aval.dtype)
    self.state = np.zeros(var.aval.shape, int)
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

  def getbox(self, box):
    assert np.shape(box) == (2, self.ndim)
    invals, outvals, primitive, params = self.eqn
    in_avals = map(get_aval, invals)
    state = self.state[box_to_slice(box)]
    to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
    # First request everything we need in order to compute `box`
    for u_box in box_finder(state, UNKNOWN):
      state[box_to_slice(u_box)] = REQUESTED
      if primitive.multiple_results:
        # TODO: pass var_idx to the backward rule
        raise NotImplementedError
      else:
        inboxes = backward_rules[primitive](
            to_global_coords(u_box), *in_avals, **params)
      for val, inbox in zip(invals, inboxes):
        if isinstance(val, LazyArray):
          val.getbox(inbox)
    # Then compute any remaining unknowns in `box`
    for u_box in box_finder(state, REQUESTED):
      if primitive.multiple_results:
        # TODO: pass var_idx to the update rule
        raise NotImplementedError
        pass
      else:
        self.cache = update_rules[primitive](
            self.cache, to_global_coords(u_box), *invals, **params)
      state[box_to_slice(u_box)] = KNOWN
    return self.cache[box_to_slice(box)]  # TODO: replace with lax.slice

def get_aval(x):
  if isinstance(x, LazyArray):
    return x.aval
  else:
    return jc.get_aval(x)

def tie_the_knot(typed_jaxpr):
  jaxpr, _, in_avals, out_avals = typed_jaxpr
  assert all(i == o for i, o in zip(in_avals, out_avals))
  in2out = dict(zip(jaxpr.invars, jaxpr.outvars))
  def replace(eqn):
    invars = [in_to_out[i] if (isinstance(i, jc.Var) and i in in_to_out) else i
              for i in eqn.invars]
    return jc.JaxprEqn(invars, eqn.outvars, eqn.primitive, eqn.params)
  eqns = [replace(eqn) for eqn in jaxpr.eqns]
  new_jaxpr = jc.Jaxpr(jaxpr.constvars, [], jaxpr.outvars, eqns)
  return jc.TypedJaxpr(new_jaxpr, typed_jaxpr.literals, [],
                       typed_jaxpr.out_avals)

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

def naryop_backward_rule(outbox, *in_avals, **params):
  return [zip(*((0, 1) if s == 1 else b
                for s, b in zip(aval.shape, zip(*outbox))))
          if aval.shape else [] for aval in in_avals]

def naryop_update_rule(op, cache, outbox, *invals, **params):
  inboxes = naryop_backward_rule(outbox, *map(get_aval, invals))
  invals = [val.getbox(box) if isinstance(val, LazyArray)
            else val[box_to_slice(box)] for val, box in zip(invals, inboxes)]
  out = op.bind(*invals, **params)
  return index_update(cache, box_to_slice(outbox), out)

backward_rules[lax.add_p] = naryop_backward_rule
update_rules[lax.add_p] = partial(naryop_update_rule, lax.add_p)

def test_boxes(starts, sizes, dim):
  assert sizes[dim] == 1
  i = 1
  while True:
    yield tuple(start + i if d == dim else slice(start, start + size)
                for d, (start, size) in enumerate(zip(starts, sizes)))
    i = i + 1

def box_finder(known, value):
  it = np.nditer(known, flags=['multi_index'])
  for k in it:
    if k == value:
      starts = it.multi_index
      sizes = known.ndim * [1]
      for d in range(known.ndim):
        box_iter = test_boxes(starts, sizes, d)
        while (starts[d] + sizes[d] < known.shape[d] and
               np.all(known[next(box_iter)] == value)):
          sizes[d] = sizes[d] + 1
      yield starts, sizes

if __name__ == "__main__":
  from jax import make_jaxpr

  x = jnp.array([1, 2, 3])
  y = jnp.array([4, 5, 6])

  jaxpr = make_jaxpr(lambda x, y: lax.add(x, y))(x, y)

  out, = lazy_eval_jaxpr(jaxpr.jaxpr, [], x, y)
