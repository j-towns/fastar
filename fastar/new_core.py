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
  __slots__ = ['cache', 'state', 'eqn', 'var_idx', 'child_counts']

  def __init__(self, var):
    self.cache = jnp.zeros(var.aval.shape, var.aval.dtype)
    self.state = np.zeros(var.aval.shape, int)
    self.child_counts = np.zeros(var.aval.shape, int)
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

  @property
  def _aval(self):
    return self.cache.aval

  def _compute_ancestral_child_counts(self, box):
    invals, outvals, primitive, params = self.eqn
    in_avals = map(get_aval, invals)
    local_state = self.state[box_to_slice(box)] if self.shape else self.state
    to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
    for u_box in box_finder(local_state, UNKNOWN):
      local_state[box_to_slice(u_box)] = REQUESTED
      if primitive.multiple_results:
        # TODO: pass var_idx to the backward rule
        raise NotImplementedError
      else:
        inboxes, counts = backward_rules[primitive](
            to_global_coords(u_box), *in_avals, **params)
        for ival, ibox, count in zip(invals, inboxes, counts):
          if isinstance(ival, LazyArray):
            ival.child_counts[box_to_slice(ibox)] += count
            ival._compute_ancestral_child_counts(ibox)

  def _toposort(self, box):
    self._compute_ancestral_child_counts(box)
    to_global_coords = lambda b: (np.add(box[0], b[0]), b[1])
    sorted_boxes = []
    local_child_counts = (self.child_counts[box_to_slice(box)] if self.shape
                          else self.child_counts)
    childless_boxes = [(self, to_global_coords(b)) for b in
                       static_box_finder(local_child_counts, 0)]
    while childless_boxes:
      arr, box = childless_boxes.pop()
      sorted_boxes.append((arr, box))
      invals, _, primitive, params = arr.eqn
      in_avals = map(get_aval, invals)
      inboxes, counts = backward_rules[primitive](box, *in_avals, **params)
      for ival, ibox, count in zip(invals, inboxes, counts):
        if isinstance(ival, LazyArray):
          to_iglobal_coords = lambda b: (np.add(ibox[0], b[0]), b[1])
          ichild_counts = (ival.child_counts[box_to_slice(ibox)] if ival.shape
                           else ival.child_counts)
          ichild_counts -= count
          childless_boxes.extend(
              [(ival, to_iglobal_coords(b))
               for b in static_box_finder(ichild_counts[box_to_slice(ibox)], 0)])
    return sorted_boxes[::-1]

  def getbox(self, box):
    assert np.shape(box) == (2, self.ndim)
    for arr, u_box in self._toposort(box):
      invals, outvals, primitive, params = arr.eqn
      if primitive.multiple_results:
        raise NotImplementedError
      else:
        invals = [val.cache if isinstance(val, LazyArray) else val
                  for val in invals]
        arr.cache = update_rules[primitive](
            arr.cache, u_box, *invals, **params)
        arr.state[box_to_slice(u_box)] = KNOWN
    return self.cache[box_to_slice(box)]

def get_aval(x):
  if isinstance(x, LazyArray):
    return x._aval
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
  out_starts, out_shape = outbox
  in_starts = [
      [0 if s == 1 else o_start for s, o_start in zip(aval.shape, out_starts)]
      if aval.shape else [] for aval in in_avals]
  in_shapes = [
      [1 if s == 1 else o_dim for s, o_dim in zip(aval.shape, out_shape)]
      if aval.shape else [] for aval in in_avals]
  in_counts = [
      np.prod([o_dim if s == 1 else 1
               for s, o_dim in zip(aval.shape, out_shape)])
      if aval.shape else np.prod(out_shape) for aval in in_avals]
  return zip(in_starts, in_shapes), in_counts

def naryop_update_rule(op, cache, outbox, *invals, **params):
  inboxes, _ = naryop_backward_rule(outbox, *map(get_aval, invals))
  invals = [val[box_to_slice(box)] for val, box in zip(invals, inboxes)]
  return lax.dynamic_update_slice(cache, op.bind(*invals, **params), outbox[0])

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

def static_box_finder(known, value):
  tmp = known == value
  ret = []
  for box in box_finder(tmp, 1):
    ret.append(box)
    tmp[box_to_slice(box)] = 0
  return list(ret)

if __name__ == "__main__":
  from jax import make_jaxpr

  x = jnp.array([1, 2, 3])
  y = jnp.array([4])
  z = jnp.array([5])

  jaxpr = make_jaxpr(lambda x, y, z: lax.add(x, lax.add(y, z)))(x, y, z)

  out, = lazy_eval_jaxpr(jaxpr.jaxpr, [], x, y, z)
