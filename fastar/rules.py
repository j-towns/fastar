from functools import partial

import numpy as np
from jax import lax, numpy as jnp
from jax.util import safe_map, safe_zip, curry, unzip2, prod, unzip3

from fastar.core import backward_rules

map = safe_map
zip = safe_zip

@curry
def naryop_backward_rule(prim, outbox, *shapes, **params):
  outstarts, outshape = outbox
  shapes = [np.array(shape) for shape in shapes]
  instarts = [np.where(shape == 1, 0, outstarts) if shape.size else []
              for shape in shapes]
  incounts = [np.full(np.where(shape == 1, 1, outshape),
                      prod(np.where(shape == 1, outshape, 1)))
              if shape.size else np.prod(outshape, dtype=int)
              for shape in shapes]
  return instarts, incounts, lambda *inslices: prim.bind(*inslices, **params)

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
def reduce_backward_rule(prim, outbox, shape, axes):
  outstart, outshape = outbox
  instart = list(outstart)
  inshape = list(outshape)
  for d in np.sort(axes):
    instart.insert(d, 0)
    inshape.insert(d, shape[d])
  return ([instart], [np.ones(inshape, int)],
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

def squeeze_backward_rule(outbox, shape, dimensions):
  outstart, outshape = outbox
  instart = list(outstart)
  inshape = list(outshape)
  for d in np.sort(dimensions):
    instart.insert(d, 0)
    inshape.insert(d, 1)
  return ([instart], [np.ones(inshape, int)],
          lambda inslice: lax.squeeze(inslice, dimensions))

backward_rules[lax.squeeze_p] = squeeze_backward_rule

def concatenate_backward_rule(outbox, *shapes, dimension):
  dim = dimension
  outstart, outshape = map(list, outbox)
  dimstart, dimshape = outstart[dim], outshape[dim]
  position = 0
  instarts = []
  incounts = []
  for shape in shapes:
    if dimstart < position + shape[dim] and position < dimstart + dimshape:
      instart = (outstart[:dim]
                 + [max(0, dimstart - position)] + outstart[dim + 1:])
      inshape = (outshape[:dim]
                 + [min(dimstart + dimshape - position, shape[dim], dimshape,
                        position + shape[dim] - instart[dim])]
                 + outshape[dim + 1:])
      instarts.append(instart)
      incounts.append(np.ones(inshape, int))
    else:
      instarts.append(None)
      incounts.append(None)
    position += shape[dim]

  return instarts, incounts, lambda *inslices: lax.concatenate(
    [x for x in inslices if x is not None], dimension)

backward_rules[lax.concatenate_p] = concatenate_backward_rule

def slice_backward_rule(outbox, _, start_indices, limit_indices, strides):
  if strides is not None:
    raise NotImplementedError('Strided slice is not yet implemented')
  outstart, outshape = outbox
  return ([np.add(outstart, start_indices)], [np.ones(outshape, int)],
          lambda inslice: inslice)

backward_rules[lax.slice_p] = slice_backward_rule

def transpose_backward_rule(outbox, _, permutation):
  outstart, outshape = outbox
  inverse_perm = np.argsort(permutation)
  return ([np.take(outstart, inverse_perm)],
          [np.ones(np.take(outshape, inverse_perm), int)],
          lambda inslice: lax.transpose(inslice, permutation))

backward_rules[lax.transpose_p] = transpose_backward_rule

def rev_backward_rule(outbox, shape, dimensions):
  outstart, outshape = outbox
  instart = [size - (start + outsize) if d in dimensions else start
             for d, (size, outsize, start)
             in enumerate(zip(shape, outshape, outstart))]
  return ([instart], [np.ones(outshape, int)],
          lambda inslice: lax.rev(inslice, dimensions))

backward_rules[lax.rev_p] = rev_backward_rule

def broadcast_in_dim_backward_rule(outbox, opshape, shape, broadcast_dimensions):
  outstart, outshape = outbox
  is_broadcast = np.array([
    d not in broadcast_dimensions or
    shape[d] != opshape[np.argwhere(np.equal(broadcast_dimensions, d)).item()]
    for d in range(len(outshape))])
  instart = np.take(outstart, broadcast_dimensions)
  inshape = np.take(np.where(is_broadcast, 1, outshape), broadcast_dimensions)
  incount = np.full(inshape, prod(np.where(is_broadcast, outshape, 1)))
  return [instart], [incount], lambda inslice: lax.broadcast_in_dim(
    inslice, outshape, broadcast_dimensions)

backward_rules[lax.broadcast_in_dim_p] = broadcast_in_dim_backward_rule

def dot_general_backward_rule(outbox, lhs_shape, rhs_shape, dimension_numbers, precision):
  out_start, out_shape = outbox
  outslices = list(zip(*outbox))
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_other_out_dims = list(range(len(lhs_batch), len(lhs_shape) - len(lhs_contracting)))
  rhs_other_out_dims = list(range(len(rhs_batch) + len(lhs_other_out_dims), len(out_shape)))
  lhs_outbox = unzip2([outslices[d] for d in list(lhs_batch) + lhs_other_out_dims])
  (lhs_start,), (lhs_incount,), _ = reduce_backward_rule(None)(lhs_outbox, lhs_shape, axes=lhs_contracting)
  rhs_outbox = unzip2([outslices[d] for d in list(rhs_batch) + rhs_other_out_dims])
  (rhs_start,), (rhs_incount,), _ = reduce_backward_rule(None)(rhs_outbox, rhs_shape, axes=rhs_contracting)
  incounts =  [lhs_incount * prod([out_shape[d] for d in rhs_other_out_dims]),
               rhs_incount * prod([out_shape[d] for d in lhs_other_out_dims])]
  return ([lhs_start, rhs_start], incounts,
          lambda *inslices: lax.dot_general(*inslices, dimension_numbers, precision))

backward_rules[lax.dot_general_p] = dot_general_backward_rule

def pad_backward_rule(outbox, shape, _, padding_config):
  outstart, outshape = outbox
  lo, _, interior = unzip3(padding_config)
  dilation = np.array(interior) + 1
  assert type(outstart) == np.ndarray
  inclip = lambda indices: np.clip(indices, 0, shape)
  lo_sign = np.where(np.less(lo, 0), -1, 1)
  instart = inclip(lo_sign * np.floor_divide(lo_sign * (outstart - lo), dilation))
  instop = inclip(lax.lax._ceil_divide(outstart + outshape - lo, dilation))
  inshape = instop - instart
  insize = prod(inshape)
  padcount = prod(outshape) - insize
  def outslice(inslice, padding_value):
    if inslice is None:
      return jnp.full(outshape, padding_value)
    next_instart = inclip(lax.lax._ceil_divide(outstart - lo, dilation))
    next_outstart = next_instart * dilation + lo
    _lo = next_outstart - outstart
    _hi = np.array(outshape) - (_lo + (jnp.array(inslice.shape) - 1) * dilation + 1)
    return lax.pad(inslice, padding_value, zip(_lo, _hi, interior))
  return ([instart if insize else None, ()],
          [np.ones(inshape, int) if insize else None, padcount], outslice)

backward_rules[lax.pad_p] = pad_backward_rule