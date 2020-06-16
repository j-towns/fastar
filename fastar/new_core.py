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

def box_to_slice(box):
  return tuple(slice(start, start + size) for start, size in box)

backward_rules = {}
update_rules = {}

class LazyArray(object):
  __slots__ = ['env', 'cache', 'known', 'primitive', 'invars', 'params',
               'invals']

  def __init__(self, aval, primitive, env, invars, params):
    assert not primitive.multiple_results, "Not implemented"
    self.cache = jnp.zeros(aval.shape, aval.dtype)
    self.known = np.zeros(aval.shape, int)
    self.primitive = primitive
    self.env = env
    self.invars = invars
    self.params = params
    self.invals = None

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
    if self.invals is None:
      self.invals = [read(self.env, var) for var in self.invars]
    invals = self.invals
    in_avals = map(get_aval, invals)
    assert np.shape(box) == (self.ndim, 2)
    # First summon everything we need in order to compute `box`
    for u_box in dynamic_box_finder(self.known, box):
      self.known[box_to_slice(u_box)] = -1
      inboxes = backward_rules[self.primitive](u_box, *in_avals, **self.params)
      for val, inbox in zip(invals, inboxes):
        if isinstance(val, LazyArray):
          val.getbox(inbox)
    # Then compute any remaining unknowns in `box`
    for u_box in box_finder(self.known, box):
      self.cache = update_rules[self.primitive](
          self.cache, u_box, *invals, **self.params)
      self.known[box_to_slice(u_box)] = 1
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

def read(env, v):
  if type(v) is jc.Literal:
    return v.val
  else:
    return env[v]

def lazy_eval_jaxpr(jaxpr, consts, *args):
  def write(v, val):
    env[v] = val

  env = {}
  write(jc.unitvar, jc.unit)
  map(write, jaxpr.constvars, consts)
  map(write, jaxpr.invars, args)
  for eqn in jaxpr.eqns:
    call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
    if call_jaxpr or eqn.primitive.multiple_results:
      raise NotImplementedError
    else:
      outvar = eqn.outvars[0]
      write(outvar, LazyArray(outvar.aval, eqn.primitive, env, eqn.invars,
                              eqn.params))
  return map(partial(read, env), jaxpr.outvars)

def naryop_backward_rule(outbox, *in_avals, **params):
  return [[(0, 1) if s == 1 else b for s, b in zip(aval.shape, outbox)]
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

def dynamic_box_finder(known, box):
  known = known[box_to_slice(box)]
  it = np.nditer(known, flags=['multi_index'])
  for k in it:
    if k == 0:
      starts = it.multi_index
      sizes = known.ndim * [1]
      for d in range(known.ndim):
        box_iter = test_boxes(starts, sizes, d)
        while (starts[d] + sizes[d] < known.shape[d] and
               np.all(known[next(box_iter)] == 0)):
          sizes[d] = sizes[d] + 1
      starts = np.add(starts, np.transpose(box)[0])
      yield zip(starts, sizes)

def box_finder(known, box):
  known = known[box_to_slice(box)] == -1  # Make a copy
  it = np.nditer(known, flags=['multi_index'])
  boxes = []
  for k in it:
    if k == True:
      starts = it.multi_index
      sizes = known.ndim * [1]
      for d in range(known.ndim):
        box_iter = test_boxes(starts, sizes, d)
        while (starts[d] + sizes[d] < known.shape[d] and
               np.all(known[next(box_iter)])):
          sizes[d] = sizes[d] + 1
      known[box_to_slice(zip(starts, sizes))] = False
      starts = np.add(starts, np.transpose(box)[0])
      boxes.append(zip(starts, sizes))
  return boxes


if __name__ == "__main__":
  from jax import make_jaxpr

  x = jnp.array([1, 2, 3])
  y = jnp.array([4, 5, 6])

  jaxpr = make_jaxpr(lambda x, y: lax.add(x, y))(x, y)

  out, = lazy_eval_jaxpr(jaxpr.jaxpr, [], x, y)
