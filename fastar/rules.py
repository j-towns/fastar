import numpy as np
from jax import numpy as jnp, lax, lax_reference as laxref, ShapedArray
from jax.util import safe_map, safe_zip, curry, unzip2, prod, unzip3

from fastar.core import dependency_rules, Ones, is_ones, materialize
from fastar.jaxpr_util import abstractify

map = safe_map
zip = safe_zip

@curry
def naryop_dependency_rule(prim, outstart, outcount, *operands, **params):
  if not is_ones(outcount):
    raise NotImplementedError
  shapes = [np.array(np.shape(o)) for o in operands]
  instarts = [np.where(shape == 1, 0, outstart) if len(shape) else []
              for shape in shapes]
  incounts = [np.full(np.where(shape == 1, 1, outcount.shape),
                      prod(np.where(shape == 1, outcount.shape, 1)))
              if len(shape) else outcount.size
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
def reduce_dependency_rule(prim, outstart, outcount, operand, axes, **kwargs):
  if not is_ones(outcount):
    raise NotImplementedError
  instart = list(outstart)
  inshape = list(outcount.shape)
  for d in np.sort(axes):
    instart.insert(d, 0)
    inshape.insert(d, operand.shape[d])
  return ([instart], [Ones(inshape)],
          lambda inslice: prim.bind(inslice, axes=axes, **kwargs))

reduce_ops = [
  lax.reduce_sum_p,
  lax.reduce_prod_p,
  lax.reduce_max_p,
  lax.reduce_min_p,
  lax.reduce_or_p,
  lax.reduce_and_p,
  lax.argmax_p,
  lax.argmin_p,
]

for op in reduce_ops:
  dependency_rules[op] = reduce_dependency_rule(op)

def squeeze_dependency_rule(outstart, outcount, operand, dimensions):
  if not is_ones(outcount):
    raise NotImplementedError
  instart = list(outstart)
  inshape = list(outcount.shape)
  for d in np.sort(dimensions):
    instart.insert(d, 0)
    inshape.insert(d, 1)
  return ([instart], [Ones(inshape)],
          lambda inslice: lax.squeeze(inslice, dimensions))

dependency_rules[lax.squeeze_p] = squeeze_dependency_rule

def concatenate_dependency_rule(outstart, outcount, *operands, dimension):
  if not is_ones(outcount):
    raise NotImplementedError
  dim = dimension
  outstart, outshape = list(outstart), list(outcount.shape)
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
      incounts.append(Ones(inshape))
    else:
      instarts.append(None)
      incounts.append(None)
    position += shape[dim]

  return instarts, incounts, lambda *inslices: lax.concatenate(
    [x for x in inslices if x is not None], dimension)

dependency_rules[lax.concatenate_p] = concatenate_dependency_rule

def slice_dependency_rule(outstart, outcount, operand, start_indices, limit_indices, strides):
  strides = np.ones_like(outcount.shape) if strides is None else np.array(strides)
  zeros = np.zeros_like(outcount.shape)
  return ([outstart * strides + start_indices],
          [laxref.pad(materialize(outcount), 0, zip(zeros, zeros, strides - 1))],
          lambda inslice: lax.slice(inslice, zeros, inslice.shape, strides))

dependency_rules[lax.slice_p] = slice_dependency_rule

def transpose_dependency_rule(outstart, outcount, operand, permutation):
  inverse_perm = np.argsort(permutation)
  return ([np.take(outstart, inverse_perm)],
          [Ones(np.take(outcount.shape, inverse_perm))
           if is_ones(outcount) else np.transpose(outcount, inverse_perm)],
          lambda inslice: lax.transpose(inslice, permutation))

dependency_rules[lax.transpose_p] = transpose_dependency_rule

def rev_dependency_rule(outstart, outcount, operand, dimensions):
  instart = [size - (start + outsize) if d in dimensions else start
             for d, (size, outsize, start)
             in enumerate(zip(operand.shape, outcount.shape, outstart))]
  return ([instart], [Ones(outcount.shape) if is_ones(outcount) else
                      lax.rev(outcount, dimensions)],
          lambda inslice: lax.rev(inslice, dimensions))

dependency_rules[lax.rev_p] = rev_dependency_rule

def broadcast_in_dim_dependency_rule(
    outstart, outcount, operand, shape, broadcast_dimensions):
  if not is_ones(outcount):
    raise NotImplementedError
  outshape = outcount.shape
  is_broadcast = np.not_equal(
      np.shape(operand), np.take(shape, broadcast_dimensions))
  instart = np.where(is_broadcast, 0, np.take(outstart, broadcast_dimensions))
  inshape = np.where(is_broadcast, 1, np.take(outshape, broadcast_dimensions))
  incount = np.full(inshape, prod(shape) // prod(operand.shape))
  return [instart], [incount], lambda inslice: lax.broadcast_in_dim(
    inslice, outshape, broadcast_dimensions)

dependency_rules[lax.broadcast_in_dim_p] = broadcast_in_dim_dependency_rule

def dot_general_dependency_rule(outstart, outcount, lhs, rhs, dimension_numbers, precision):
  if not is_ones(outcount):
    raise NotImplementedError
  outshape = outcount.shape
  outslices = list(zip(outstart, outshape))
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_other_out_dims = list(range(len(lhs_batch), len(lhs.shape) - len(lhs_contracting)))
  rhs_other_out_dims = list(range(len(rhs_batch) + len(lhs_other_out_dims), len(outshape)))
  lhs_outstart, lhs_outshape = unzip2([outslices[d] for d in list(lhs_batch) + lhs_other_out_dims])
  (lhs_start,), (lhs_count,), _ = reduce_dependency_rule(None)(lhs_outstart, Ones(lhs_outshape), lhs, axes=lhs_contracting)
  rhs_outstart, rhs_outshape = unzip2([outslices[d] for d in list(rhs_batch) + rhs_other_out_dims])
  (rhs_start,), (rhs_count,), _ = reduce_dependency_rule(None)(rhs_outstart, Ones(rhs_outshape), rhs, axes=rhs_contracting)
  incounts = [materialize(lhs_count) * prod([outshape[d] for d in rhs_other_out_dims]),
              materialize(rhs_count) * prod([outshape[d] for d in lhs_other_out_dims])]
  return ([lhs_start, rhs_start], incounts,
          lambda *inslices: lax.dot_general(*inslices, dimension_numbers, precision))

dependency_rules[lax.dot_general_p] = dot_general_dependency_rule

def pad_dependency_rule(outstart, outcount, operand, padding_value, padding_config):
  lo, _, interior = unzip3(padding_config)
  dilation = np.array(interior) + 1
  outstart_lo = np.subtract(outstart, lo)
  inclip = lambda indices: np.clip(indices, 0, operand.shape)
  instart = inclip(lax.lax._ceil_divide(outstart_lo, dilation))
  instop = inclip(lax.lax._ceil_divide(outstart_lo + outcount.shape, dilation))
  inshape = instop - instart
  insize = prod(inshape)
  offset = instart * dilation - outstart_lo
  limit = offset + np.maximum(0, (np.array(inshape) - 1) * dilation + 1)
  incount = Ones(inshape) if is_ones(outcount) else laxref.slice(
    outcount, offset, limit, dilation) if insize else None
  padcount = outcount.size - insize
  def outslice(inslice, padding_value):
    assert inslice is None or np.array_equal(inslice.shape, inshape)
    return (lax.pad(inslice, padding_value,
                    zip(offset, np.array(outcount.shape) - limit, interior))
            if insize else jnp.full(outcount.shape, padding_value, operand.dtype))
  return ([instart if insize else None, ()],
          [incount, padcount], outslice)

dependency_rules[lax.pad_p] = pad_dependency_rule

def outer_product(vectors):
  return np.einsum(*(x for (i, vector) in enumerate(vectors) for x in (vector, [i])))

def conv_incounts(lhs_shape, rhs_shape, window_strides):
  batch_size, _, *spatial_lhs_shape = lhs_shape
  outchannel_size, _, *spatial_rhs_shape = rhs_shape
  single_dim_counts = []
  for size, rsize, stride in zip(spatial_lhs_shape, spatial_rhs_shape,
                                 window_strides):
    strides_per_tile = lax.lax._ceil_divide(rsize, stride)
    lo = np.arange(strides_per_tile) * stride
    tile_size = strides_per_tile * stride
    tile_counts = np.floor_divide(size - lo + (tile_size - rsize), tile_size)
    tile = np.concatenate([np.ones(rsize, int), np.zeros(tile_size - rsize, int)])
    hi = size - lo - tile_size * tile_counts
    counts = [laxref.pad(np.tile(tile, (tile_count,)), 0, ((lo, hi, 0),))
              for lo, hi, tile_count in zip(lo, hi, tile_counts)]
    single_dim_counts.append(np.sum(counts, axis=0))
  lhs_count = np.broadcast_to(
    outer_product(single_dim_counts) * outchannel_size, lhs_shape)
  return lhs_count, np.full(rhs_shape, batch_size)

def conv_dependency_rule(outstart, outcount, lhs, rhs, window_strides, precision):
  if not is_ones(outcount):
    raise NotImplementedError
  batch_start, outchannel_start, *spatial_outstart = outstart
  batch_size, outchannel_size, *spatial_outshape = outcount.shape
  lhs_start = [batch_start, 0] + list(np.array(spatial_outstart) * window_strides)
  lhs_shape = [batch_size, lhs.shape[1]] + list(np.subtract(spatial_outshape, 1) * window_strides + rhs.shape[2:])
  full_rhs_channels = list(rhs.shape[1:])
  rhs_start = [outchannel_start] + [0] * len(full_rhs_channels)
  rhs_shape = [outchannel_size] + full_rhs_channels
  return ((lhs_start, rhs_start), conv_incounts(lhs_shape, rhs_shape, window_strides),
          lambda lhs_slice, rhs_slice: lax.conv(lhs_slice, rhs_slice, window_strides, 'VALID', precision))

def conv_general_dilated_dependency_rule(
    outstart, outcount, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, feature_group_count, batch_group_count, precision, **_):
  if not is_ones(outcount):
    raise NotImplementedError
  if feature_group_count > 1 or batch_group_count > 1:
    raise NotImplementedError("Feature and batch groups are not implemented.")
  if np.any(np.not_equal(rhs_dilation, 1)):
    raise NotImplementedError("Dilated convs are not yet implemented.")
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  # abstract evaluations to retrieve shapes:
  lhs = lax.transpose_p.abstract_eval(abstractify(lhs), permutation=lhs_spec)
  rhs = lax.transpose_p.abstract_eval(abstractify(rhs), permutation=rhs_spec)
  pad_val = ShapedArray((), lhs.dtype)
  pad_config = [(0, 0, 0)] * 2 + [
    (lo, hi, dil - 1) for (lo, hi), dil in zip(padding, lhs_dilation)]
  padded_lhs = lax.pad_p.abstract_eval(lhs, pad_val, padding_config=pad_config)
  (outstart,), (outcount,), out_transpose = transpose_dependency_rule(
    outstart, outcount, None, np.argsort(out_spec))
  (padded_lhs_start, rhs_start), (padded_lhs_count, rhs_count), conv = conv_dependency_rule(
    outstart, outcount, padded_lhs, rhs, window_strides, precision)
  (lhs_start, _), (lhs_count, _), lhs_pad = pad_dependency_rule(
    padded_lhs_start, padded_lhs_count, lhs, pad_val, pad_config)
  if lhs_start is not None:
    (lhs_start,), (lhs_count,), lhs_transpose = transpose_dependency_rule(
      lhs_start, lhs_count, None, lhs_spec)
  (rhs_start,), (rhs_count,), rhs_transpose = transpose_dependency_rule(
    rhs_start, rhs_count, None, rhs_spec)
  return ([lhs_start, rhs_start], [lhs_count, rhs_count],
          lambda lhs_slice, rhs_slice: out_transpose(conv(
            lhs_pad(None if lhs_slice is None else lhs_transpose(lhs_slice),
                    np.zeros((), lhs.dtype)),
            rhs_transpose(rhs_slice))))

dependency_rules[lax.conv_general_dilated_p] = conv_general_dilated_dependency_rule
