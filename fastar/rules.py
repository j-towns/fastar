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

naryops = [
    lax.lt_p,
    lax.le_p,
    lax.gt_p,
    lax.ge_p,
    lax.ne_p,
    lax.eq_p,
    lax.shift_right_logical_p,
    lax.shift_right_arithmetic_p,
    lax.shift_left_p,
    lax.min_p,
    lax.max_p,
    lax.rem_p,
    lax.div_p,
    lax.mul_p,
    lax.sub_p,
    lax.add_p,
    lax.population_count_p,
    lax.xor_p,
    lax.or_p,
    lax.and_p,
    lax.not_p,
    lax.pow_p,
    lax.rsqrt_p,
    lax.sqrt_p,
    lax.abs_p,
    lax.conj_p,
    lax.complex_p,
    lax.imag_p,
    lax.real_p,
    lax.erf_inv_p,
    lax.erfc_p,
    lax.erf_p,
    lax.bessel_i1e_p,
    lax.bessel_i0e_p,
    lax.igammac_p,
    lax.igamma_grad_a_p,
    lax.igamma_p,
    lax.digamma_p,
    lax.lgamma_p,
    lax.regularized_incomplete_beta_p,
    lax.atanh_p,
    lax.acosh_p,
    lax.asinh_p,
    lax.cosh_p,
    lax.sinh_p,
    lax.atan2_p,
    lax.cos_p,
    lax.sin_p,
    lax.tanh_p,
    lax.log1p_p,
    lax.expm1_p,
    lax.log_p,
    lax.exp_p,
    lax.is_finite_p,
    lax.round_p,
    lax.ceil_p,
    lax.floor_p,
    lax.nextafter_p,
    lax.sign_p,
    lax.neg_p,
    lax.select_p,
]

for op in naryops:
  backward_rules[op] = naryop_backward_rule
  update_rules[op] = partial(naryop_update_rule, op)

def concatenate_backward_rule(outbox, *invals, **params):
  # from IPython.terminal.debugger import set_trace; set_trace()
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
                 + [min(dimstart + dimshape - position, v_shape[dim], dimshape,
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

def broadcast_in_dim_backward_rule(
    outbox, operand, shape, broadcast_dimensions):
  pass
