from functools import partial

import numpy as np
from jax import lax
from jax.util import safe_map, safe_zip
import jax.numpy as jnp

from fastar.core import backward_rules, update_rules, get_aval
from fastar.box_util import box_to_slice


map = safe_map
zip = safe_zip

def naryop_backward_rule(outbox, *invals, **params):
  out_starts, out_shape = outbox
  in_starts = [
      [0 if s == 1 else o_start
       for s, o_start in zip(np.shape(val), out_starts)]
      if np.shape(val) else [] for val in invals]
  in_shapes = [
      [1 if s == 1 else o_dim for s, o_dim in zip(np.shape(val), out_shape)]
      if np.shape(val) else [] for val in invals]
  in_counts = [
      np.prod([o_dim if s == 1 else 1
               for s, o_dim in zip(np.shape(val), out_shape)])
      if np.shape(val) else np.prod(out_shape) for val in invals]
  in_counts = map(partial(np.full, dtype=int), in_shapes, in_counts)
  return in_starts, in_counts

def naryop_update_rule(op, cache, outbox, *invals, **params):
  instarts, incounts = naryop_backward_rule(outbox, *map(get_aval, invals))
  inboxes = [(start, count.shape) for start, count in zip(instarts, incounts)]
  invals = [val if jnp.isscalar(val) else val[box_to_slice(box)]
            for val, box in zip(invals, inboxes)]
  return lax.dynamic_update_slice(cache, op.bind(*invals, **params), outbox[0])

backward_rules[lax.add_p] = naryop_backward_rule
update_rules[lax.add_p] = partial(naryop_update_rule, lax.add_p)

def concatenate_backward_rule(outbox, *invals, **params):
  dim = params['dimension']
  outstart, outshape = map(list, outbox)
  dimstart, dimshape  = outstart[dim], outshape[dim]
  position = 0
  instarts = []
  incounts = []
  for v in invals:
    v_shape = np.shape(v)
    if dimstart < position + v_shape[dim] and position < dimstart + dimshape:
      instart = (outstart[:dim]
                 + [max(0, dimstart - position)] + outstart[dim + 1:])
      inshape = (outshape[:dim]
                 + [min(dimstart + dimshape - position, v_shape[dim],
                        position + v_shape[dim] - instart[dim])]
                 + outshape[dim + 1:])
      instarts.append(instart)
      incounts.append(np.ones(inshape, int))
    else:
      instarts.append(None)
      incounts.append(None)
    position = position + v_shape[dim]
  return instarts, incounts

def concatenate_update_rule(cache, outbox, *invals, **params):
  dim = params['dimension']
  instarts, incounts = concatenate_backward_rule(outbox, *invals, **params)
  inboxes = [None if start is None else (start, count.shape)
             for start, count in zip(instarts, incounts)]
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
  return [in_start], [np.ones(out_shape, int)]

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
