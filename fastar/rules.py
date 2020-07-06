from functools import partial

import numpy as np
from jax import lax, numpy as jnp
from jax.util import safe_map, safe_zip, curry, unzip2, unzip3

from fastar.core import dependency_rules

map = safe_map
zip = safe_zip

def tie_in_dependency_rule(outbox, x, y):
  outstarts, outshape = outbox
  instarts = [None, outstarts]
  incounts = [None, np.ones(outshape, int)]
  return instarts, incounts, lambda x_part, y_part: lax.tie_in(x, y_part)

dependency_rules[lax.tie_in_p] = tie_in_dependency_rule

@curry
def naryop_dependency_rule(prim, outbox, *operands, **params):
  outstarts, outshape = outbox
  shapes = [np.array(np.shape(o)) for o in operands]
  instarts = [np.where(shape == 1, 0, outstarts) if len(shape) else []
              for shape in shapes]
  incounts = [np.full(np.where(shape == 1, 1, outshape),
                      np.prod(np.where(shape == 1, outshape, 1)))
              if len(shape) else np.prod(outshape, dtype=int)
              for shape in shapes]
  return instarts, incounts, lambda *inslices: prim.bind(*inslices, **params)

naryops = [
    lax.convert_element_type_p,
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
  dependency_rules[op] = naryop_dependency_rule(op)

@curry
def reduce_dependency_rule(prim, outbox, operand, axes):
  outstart, outshape = outbox
  instart = list(outstart)
  inshape = list(outshape)
  for d in np.sort(axes):
    instart.insert(d, 0)
    inshape.insert(d, operand.shape[d])
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
  dependency_rules[op] = reduce_dependency_rule(op)

def squeeze_dependency_rule(outbox, _, dimensions):
  outstart, outshape = outbox
  instart = list(outstart)
  inshape = list(outshape)
  for d in np.sort(dimensions):
    instart.insert(d, 0)
    inshape.insert(d, 1)
  return ([instart], [np.ones(inshape, int)],
          lambda inslice: lax.squeeze(inslice, dimensions))

dependency_rules[lax.squeeze_p] = squeeze_dependency_rule

def concatenate_dependency_rule(outbox, *operands, dimension):
  dim = dimension
  outstart, outshape = map(list, outbox)
  dimstart, dimshape = outstart[dim], outshape[dim]
  position = 0
  instarts = []
  incounts = []
  for operand in operands:
    shape = operand.shape
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

dependency_rules[lax.concatenate_p] = concatenate_dependency_rule

def slice_dependency_rule(outbox, _, start_indices, limit_indices, strides):
  outstart, outshape = outbox
  strides = np.ones_like(outshape) if strides is None else np.array(strides)
  zeros = np.zeros_like(outshape)
  return ([outstart * strides + start_indices],
          [lax.pad(np.ones(outshape, int), 0, zip(zeros, zeros, strides - 1))],
          lambda inslice: lax.slice(inslice, zeros, inslice.shape, strides))

dependency_rules[lax.slice_p] = slice_dependency_rule

def transpose_dependency_rule(outbox, _, permutation):
  outstart, outshape = outbox
  inverse_perm = np.argsort(permutation)
  return ([np.take(outstart, inverse_perm)],
          [np.ones(np.take(outshape, inverse_perm), int)],
          lambda inslice: lax.transpose(inslice, permutation))

dependency_rules[lax.transpose_p] = transpose_dependency_rule

def rev_dependency_rule(outbox, operand, dimensions):
  outstart, outshape = outbox
  instart = [size - (start + outsize) if d in dimensions else start
             for d, (size, outsize, start)
             in enumerate(zip(operand.shape, outshape, outstart))]
  return ([instart], [np.ones(outshape, int)],
          lambda inslice: lax.rev(inslice, dimensions))

dependency_rules[lax.rev_p] = rev_dependency_rule

def broadcast_in_dim_dependency_rule(
    outbox, operand, shape, broadcast_dimensions):
  outstart, outshape = outbox
  is_broadcast = np.not_equal(
      np.shape(operand), np.take(shape, broadcast_dimensions))
  instart = np.where(is_broadcast, 0, np.take(outstart, broadcast_dimensions))
  inshape = np.where(is_broadcast, 1, np.take(outshape, broadcast_dimensions))
  incount = np.full(inshape, np.prod(shape) // np.prod(operand.shape))
  return [instart], [incount], lambda inslice: lax.broadcast_in_dim(
    inslice, outshape, broadcast_dimensions)

dependency_rules[lax.broadcast_in_dim_p] = broadcast_in_dim_dependency_rule

def dot_general_dependency_rule(outbox, lhs, rhs, dimension_numbers, precision):
  out_start, out_shape = outbox
  outslices = list(zip(*outbox))
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_other_out_dims = list(range(len(lhs_batch), len(lhs.shape) - len(lhs_contracting)))
  rhs_other_out_dims = list(range(len(rhs_batch) + len(lhs_other_out_dims), len(out_shape)))
  lhs_outbox = unzip2([outslices[d] for d in list(lhs_batch) + lhs_other_out_dims])
  (lhs_start,), (lhs_incount,), _ = reduce_dependency_rule(None)(lhs_outbox, lhs, axes=lhs_contracting)
  rhs_outbox = unzip2([outslices[d] for d in list(rhs_batch) + rhs_other_out_dims])
  (rhs_start,), (rhs_incount,), _ = reduce_dependency_rule(None)(rhs_outbox, rhs, axes=rhs_contracting)
  incounts =  [lhs_incount * np.prod([out_shape[d] for d in rhs_other_out_dims]),
               rhs_incount * np.prod([out_shape[d] for d in lhs_other_out_dims])]
  return ([lhs_start, rhs_start], incounts,
          lambda *inslices: lax.dot_general(*inslices, dimension_numbers, precision))

dependency_rules[lax.dot_general_p] = dot_general_dependency_rule

def pad_dependency_rule(outbox, operand, _, padding_config):
  outstart, outshape = outbox
  lo, _, interior = unzip3(padding_config)
  dilation = np.array(interior) + 1
  assert type(outstart) == np.ndarray
  inclip = lambda indices: np.clip(indices, 0, operand.shape)
  lo_sign = np.where(np.less(lo, 0), -1, 1)
  instart = inclip(lo_sign * np.floor_divide(lo_sign * (outstart - lo), dilation))
  instop = inclip(lax.lax._ceil_divide(outstart + outshape - lo, dilation))
  inshape = instop - instart
  insize = np.prod(inshape)
  padcount = np.prod(outshape) - insize
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

dependency_rules[lax.pad_p] = pad_dependency_rule
