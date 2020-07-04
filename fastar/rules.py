from functools import partial

import numpy as np
from jax import lax, numpy as jnp
from jax.util import safe_map, safe_zip, curry, unzip2, prod, unzip3

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
                      prod(np.where(shape == 1, outshape, 1)))
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

def squeeze_dependency_rule(outbox, operand, dimensions):
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

def slice_dependency_rule(outbox, operand, start_indices, limit_indices, strides):
  outstart, outshape = outbox
  strides = np.ones_like(outshape) if strides is None else np.array(strides)
  zeros = np.zeros_like(outshape)
  return ([outstart * strides + start_indices],
          [lax.pad(np.ones(outshape, int), 0, zip(zeros, zeros, strides - 1))],
          lambda inslice: lax.slice(inslice, zeros, inslice.shape, strides))

dependency_rules[lax.slice_p] = slice_dependency_rule

def transpose_dependency_rule(outbox, operand, permutation):
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
  incount = np.full(inshape, prod(shape) // prod(operand.shape))
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
  incounts =  [lhs_incount * prod([out_shape[d] for d in rhs_other_out_dims]),
               rhs_incount * prod([out_shape[d] for d in lhs_other_out_dims])]
  return ([lhs_start, rhs_start], incounts,
          lambda *inslices: lax.dot_general(*inslices, dimension_numbers, precision))

dependency_rules[lax.dot_general_p] = dot_general_dependency_rule

def pad_dependency_rule(outbox, operand, padding_value, padding_config, allow_empty_slices=False):
  outstart, outshape = outbox
  lo, _, interior = unzip3(padding_config)
  dilation = np.array(interior) + 1
  outstart = np.array(outstart)
  inclip = lambda indices: np.clip(indices, 0, operand.shape)
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
  return ([instart if insize or allow_empty_slices else None, ()],
          [np.ones(inshape, int) if insize or allow_empty_slices else None, padcount], outslice)

dependency_rules[lax.pad_p] = pad_dependency_rule

def conv_lhs_count(instart, inshape, lhs_shape, rhs_shape, window_strides):
  single_dim_counts = []
  for size, rsize, stride in zip(lhs_shape[2:], rhs_shape[2:], window_strides):
    strides_per_tile = lax.lax._ceil_divide(rsize, stride)
    lo = np.arange(strides_per_tile) * stride
    tile_size = strides_per_tile * stride
    tile_counts = np.floor_divide(size - lo + (tile_size - rsize), tile_size)
    tile = np.concatenate([np.ones(rsize, int), np.zeros(tile_size - rsize, int)])
    hi = size - lo - tile_size * tile_counts
    counts = [lax.pad(np.tile(tile, (tile_count,)), 0, ((lo, hi, 0),))
              for lo, hi, tile_count in zip(lo, hi, tile_counts)]
    single_dim_counts.append(np.sum(counts, axis=0))

  single_dim_count_slices = [
    side[start: start + size]
    for side, start, size in zip(single_dim_counts, instart[2:], inshape[2:])]

  count = single_dim_count_slices[0]
  for s in single_dim_count_slices[1:]:
    count = np.outer(count, s)
  return np.broadcast_to(count, inshape)

def conv_dependency_rule(outbox, lhs, rhs, window_strides, precision):
  outstart, outshape = outbox
  batch_start, outchannel_start, *spatial_outstart = outstart
  batch_shape, outchannel_shape, *spatial_outshape = outshape
  lhs_start = [batch_start, 0] + list(np.array(spatial_outstart) * window_strides)
  lhs_shape = [batch_shape, lhs.shape[1]] + list(np.subtract(spatial_outshape, 1) * window_strides + rhs.shape[2:])
  full_rhs_channels = list(rhs.shape[1:])
  rhs_start = [outchannel_start] + [0] * len(full_rhs_channels)
  rhs_shape = [outchannel_shape] + full_rhs_channels
  def outslice(lhs_slice, rhs_slice):
    return lax.conv(lhs_slice, rhs_slice, window_strides, 'VALID', precision)
  return ((lhs_start, rhs_start),
          (conv_lhs_count(lhs_start, lhs_shape, lhs.shape, rhs.shape, window_strides),
           np.ones(rhs_shape, int)), outslice)

def conv_general_dilated_dependency_rule(
    outbox, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, feature_group_count, batch_group_count, precision, **_):
  if feature_group_count > 1 or batch_group_count > 1:
    raise NotImplementedError("Feature and batch groups are not implemented.")
  if np.any(np.not_equal(lhs_dilation, 1)):
    raise NotImplementedError("Transposed convs are not yet implemented.")
  if np.any(np.not_equal(rhs_dilation, 1)):
    raise NotImplementedError("Dilated convs are not yet implemented.")
  _, outshape = outbox

  lhs_spec, rhs_spec, out_spec = dimension_numbers
  lhs_transpose = lambda lhs: lax.transpose(lhs, lhs_spec)
  rhs_transpose = lambda rhs: lax.transpose(rhs, rhs_spec)
  # evaluations on abstract inputs to retrieve shapes:
  transposed_lhs = lhs_transpose(lhs)
  transposed_rhs = rhs_transpose(rhs)
  padding_value = np.zeros((), lhs.dtype)
  padding_args = transposed_lhs, padding_value, [(0, 0, 0)] * 2 + [(lo, hi, 0) for lo, hi in padding]
  padded_lhs = lax.pad(*padding_args)

  (outstart,), (outcount,), out_transpose = transpose_dependency_rule(outbox, None, np.argsort(out_spec))
  (padded_lhs_start, transposed_rhs_start), (padded_lhs_count, transposed_rhs_count), conv = conv_dependency_rule(
    (outstart, outcount.shape), padded_lhs, transposed_rhs, window_strides, precision)
  (transposed_lhs_start, _), (transposed_lhs_count, _), lhs_pad = pad_dependency_rule(
    (padded_lhs_start, padded_lhs_count.shape), *padding_args, allow_empty_slices=True)
  start_below_lo = np.array([lo for lo, hi in padding]) - padded_lhs_start[2:]
  start = np.concatenate([[0, 0], np.clip(start_below_lo, 0, padded_lhs_count.shape[2:])])
  inverse_lhs_perm = np.argsort(lhs_spec)
  inverse_rhs_perm = np.argsort(rhs_spec)
  lhs_start = np.take(transposed_lhs_start, inverse_lhs_perm)
  lhs_count = lax.transpose(lax.slice(
    padded_lhs_count, start, start + transposed_lhs_count.shape,
    np.ones_like(start)), inverse_lhs_perm)
  rhs_start = np.take(transposed_rhs_start, inverse_rhs_perm)
  rhs_count = lax.transpose(transposed_rhs_count, inverse_rhs_perm)
  def outslice(lhs_slice, rhs_slice):
    return out_transpose(conv(lhs_pad(
      lhs_transpose(lhs_slice), padding_value), rhs_transpose(rhs_slice)))
  return ((lhs_start, rhs_start), (lhs_count, rhs_count), outslice)

dependency_rules[lax.conv_general_dilated_p] = conv_general_dilated_dependency_rule
