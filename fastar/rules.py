from functools import partial

import numpy as np
from jax import lax
from jax.util import safe_map, safe_zip
import jax.numpy as jnp

from fastar.core import backward_rules, update_rules, get_aval
from fastar.box_util import box_to_slice


map = safe_map
zip = safe_zip

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
  invals = [val if jnp.isscalar(val) else val[box_to_slice(box)]
            for val, box in zip(invals, inboxes)]
  return lax.dynamic_update_slice(cache, op.bind(*invals, **params), outbox[0])

backward_rules[lax.add_p] = naryop_backward_rule
update_rules[lax.add_p] = partial(naryop_update_rule, lax.add_p)

def concatenate_backward_rule(outbox, *in_avals, **params):
  dim = params['dimension']
  outstart, outshape = map(list, outbox)
  dimstart, dimshape  = outstart[dim], outshape[dim]
  position = 0
  inboxes = []
  incounts = []
  for a in in_avals:
    if dimstart < position + a.shape[dim] and position < dimstart + dimshape:
      instart = (outstart[:dim]
                 + [max(0, dimstart - position)] + outstart[dim + 1:])
      inshape = (outshape[:dim]
                 + [min(position + a.shape[dim] - dimstart, a.shape[dim],
                        dimshape)]
                 + outshape[dim + 1:])
      inboxes.append((instart, inshape))
      incounts.append(np.ones(inshape, int))
    else:
      inboxes.append(None)
      incounts.append(None)
    position = position + a.shape[dim]
  return inboxes, incounts

def concatenate_update_rule(cache, outbox, *invals, **params):
  dim = params['dimension']
  inboxes, _ = concatenate_backward_rule(outbox, *invals, **params)
  invals = [val[box_to_slice(box)] for val, box in zip(invals, inboxes)
            if box is not None]
  out_start, _ = outbox
  return lax.dynamic_update_slice(
      cache, lax.concatenate(invals, dim), out_start)

backward_rules[lax.concatenate_p] = concatenate_backward_rule
update_rules[lax.concatenate_p] = concatenate_update_rule

def slice_backward_rule(outbox, operand, start_indices, limit_indices, strides):
  if strides is not None:
    raise NotImplementedError
  out_start, out_shape = outbox
  in_start = np.add(out_start, start_indices)
  return [(in_start, out_shape)], [np.ones(out_shape, int)]

def slice_update_rule(cache, outbox, operand, start_indices, limit_indices,
                      strides):
  if strides is not None:
    raise NotImplementedError
  out_start, out_shape = outbox
  in_start = np.add(out_start, start_indices)
  return lax.dynamic_update_slice(
      cache, lax.dynamic_slice(operand, in_start, out_shape), out_start)

backward_rules[lax.slice_p] = slice_backward_rule
update_rules[lax.slice_p] = slice_update_rule
