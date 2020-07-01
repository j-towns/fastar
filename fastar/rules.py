from functools import partial

import numpy as np
from jax import lax, numpy as jnp
from jax.util import safe_map, safe_zip, curry, unzip2, prod, unzip3

from fastar.core import backward_rules

map = safe_map
zip = safe_zip

@curry
def naryop_backward_rule(prim, outbox, *invals, **params):
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
  return in_starts, in_counts, lambda *inslices: prim.bind(*inslices, **params)

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
    lax.integer_pow_p,
]

for op in naryops:
  backward_rules[op] = naryop_backward_rule(op)

@curry
def reduce_backward_rule(prim, outbox, operand, axes):
  out_start, out_shape = outbox
  in_start = list(out_start)
  in_shape = list(out_shape)
  for d in np.sort(axes):
    in_start.insert(d, 0)
    in_shape.insert(d, operand.shape[d])
  return ([in_start], [np.ones(in_shape, int)],
          lambda inslice: prim.bind(inslice, axes=axes))

reduce_ops = [
  lax.reduce_sum_p,
  lax.reduce_prod_p,
  lax.reduce_max_p,
  lax.reduce_min_p,
  lax.reduce_or_p,
  lax.reduce_and_p,
]

for op in reduce_ops:
  backward_rules[op] = reduce_backward_rule(op)

def squeeze_backward_rule(outbox, operand, dimensions):
  out_start, out_shape = outbox
  in_start = list(out_start)
  in_shape = list(out_shape)
  for d in np.sort(dimensions):
    in_start.insert(d, 0)
    in_shape.insert(d, 1)
  return ([in_start], [np.ones(in_shape, int)],
          lambda inslice: lax.squeeze(inslice, dimensions))

backward_rules[lax.squeeze_p] = squeeze_backward_rule

def concatenate_backward_rule(outbox, *invals, dimension):
  dim = dimension
  outstart, outshape = map(list, outbox)
  dimstart, dimshape = outstart[dim], outshape[dim]
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

  return instarts, incounts, lambda *inslices: lax.concatenate(
    [x for x in inslices if x is not None], dimension)

backward_rules[lax.concatenate_p] = concatenate_backward_rule

def slice_backward_rule(outbox, operand, start_indices, limit_indices, strides):
  if strides is not None:
    raise NotImplementedError('Strided slice is not yet implemented')
  out_start, out_shape = outbox
  in_start = np.add(out_start, start_indices)
  return [in_start], [np.ones(out_shape, int)], lambda inslice: inslice

backward_rules[lax.slice_p] = slice_backward_rule

def transpose_backward_rule(outbox, operand, permutation):
  out_start, out_shape = outbox
  inverse_perm = np.argsort(permutation)
  return ([np.take(out_start, inverse_perm)],
          [np.ones(np.take(out_shape, inverse_perm), int)],
          lambda inslice: lax.transpose(inslice, permutation))

backward_rules[lax.transpose_p] = transpose_backward_rule

def rev_backward_rule(outbox, operand, dimensions):
  out_start, out_shape = outbox
  in_start = [op_size - (start + size) if d in dimensions else start
              for d, (op_size, size, start)
              in enumerate(zip(operand.shape, out_shape, out_start))]
  return ([in_start], [np.ones(out_shape, int)],
          lambda inslice: lax.rev(inslice, dimensions))

backward_rules[lax.rev_p] = rev_backward_rule

def broadcast_in_dim_backward_rule(
    outbox, operand, shape, broadcast_dimensions):
  out_start, out_shape = outbox
  in_start = []
  in_shape = []
  factor = 1
  for d in range(len(out_start)):
    if d in broadcast_dimensions:
      in_start.append(out_start[d])
      (in_dim,), = np.argwhere(np.equal(broadcast_dimensions, d))
      in_size = operand.shape[in_dim]
      is_broadcast = shape[d] != in_size
      if is_broadcast:
        assert in_size == 1
        factor *= out_shape[d]
      in_shape.append(1 if is_broadcast else out_shape[d])
    else:
      factor *= out_shape[d]

  return ([in_start], [np.ones(in_shape, int) * factor],
          lambda inslice: lax.broadcast_in_dim(inslice, out_shape, broadcast_dimensions))

backward_rules[lax.broadcast_in_dim_p] = broadcast_in_dim_backward_rule

def dot_general_backward_rule(outbox, lhs, rhs, dimension_numbers, precision):
  out_start, out_shape = outbox
  outslices = list(zip(*outbox))
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_other_out_dims = list(range(len(lhs_batch), len(lhs.shape) - len(lhs_contracting)))
  rhs_other_out_dims = list(range(len(rhs_batch) + len(lhs_other_out_dims), len(out_shape)))
  lhs_outbox = unzip2([outslices[d] for d in list(lhs_batch) + lhs_other_out_dims])
  (lhs_start,), (lhs_incount,), _ = reduce_backward_rule(None)(lhs_outbox, lhs, axes=lhs_contracting)
  lhs_incount *= prod([out_shape[d] for d in rhs_other_out_dims])
  rhs_outbox = unzip2([outslices[d] for d in list(rhs_batch) + rhs_other_out_dims])
  (rhs_start,), (rhs_incount,), _ = reduce_backward_rule(None)(rhs_outbox, rhs, axes=rhs_contracting)
  rhs_incount *= prod([out_shape[d] for d in lhs_other_out_dims])
  return ([lhs_start, rhs_start], [lhs_incount, rhs_incount],
          lambda *inslices: lax.dot_general(*inslices, dimension_numbers, precision))

backward_rules[lax.dot_general_p] = dot_general_backward_rule

def pad_backward_rule(outbox, operand, padding_value, padding_config):
  out_start, out_shape = outbox
  lo, _, interior = unzip3(padding_config)
  dilation = np.array(interior) + 1
  assert type(out_start) == np.ndarray
  inclip = lambda indices: np.clip(indices, 0, operand.shape)
  lo_sign = np.where(np.less(lo, 0), -1, 1)
  instart = inclip(lo_sign * np.floor_divide(lo_sign * (out_start - lo), dilation))
  instop = inclip(lax.lax._ceil_divide(out_start + out_shape - lo, dilation))
  inshape = instop - instart
  insize = prod(inshape)
  padcount = prod(out_shape) - insize
  def outslice(inslice, padding_value):
    if inslice is None:
      return jnp.full(out_shape, padding_value)
    next_instart = inclip(lax.lax._ceil_divide(out_start - lo, dilation))
    next_outstart = next_instart * dilation + lo
    _lo = next_outstart - out_start
    _hi = np.array(out_shape) - (_lo + (jnp.array(inslice.shape) - 1) * dilation + 1)
    return lax.pad(inslice, padding_value, zip(_lo, _hi, interior))
  return ([instart if insize else None, ()],
          [np.ones(inshape) if insize else None, padcount], outslice)

backward_rules[lax.pad_p] = pad_backward_rule