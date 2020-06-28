from functools import partial

import numpy as np
from jax import lax, numpy as jnp, ops
from jax.util import safe_map, safe_zip, curry, unzip2, prod, unzip3

from fastar.core import backward_rules, update_rules


map = safe_map
zip = safe_zip

@curry
def default_update_rule(backward_rule, fun, cache, outbox, *args, **kwargs):
  out_start, out_shape = outbox
  instarts, incounts = backward_rule(outbox, *args, **kwargs)
  inslices = [None if instart is None else
              lax.dynamic_slice(arg, instart, incount.shape)
              for arg, instart, incount in zip(args, instarts, incounts)]
  out_slice = fun(*inslices, **kwargs)
  return lax.dynamic_update_slice(cache, out_slice, out_start)

def def_standard_op(op, backward_rule):
  backward_rules[op] = backward_rule
  update_rules[op] = default_update_rule(backward_rule, op.bind)

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
  def_standard_op(op, naryop_backward_rule)

def reduce_backward_rule(outbox, operand, axes):
  out_start, out_shape = outbox
  in_start = list(out_start)
  in_shape = list(out_shape)
  for d in np.sort(axes):
    in_start.insert(d, 0)
    in_shape.insert(d, operand.shape[d])
  return [in_start], [np.ones(in_shape, int)]

reduce_ops = [
  lax.reduce_sum_p,
  lax.reduce_prod_p,
  lax.reduce_max_p,
  lax.reduce_min_p,
  lax.reduce_or_p,
  lax.reduce_and_p,
]

for op in reduce_ops:
  def_standard_op(op, reduce_backward_rule)

def squeeze_backward_rule(outbox, operand, dimensions):
  out_start, out_shape = outbox
  in_start = list(out_start)
  in_shape = list(out_shape)
  for d in np.sort(dimensions):
    in_start.insert(d, 0)
    in_shape.insert(d, 1)
  return [in_start], [np.ones(in_shape, int)]

def_standard_op(lax.squeeze_p, squeeze_backward_rule)

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

backward_rules[lax.concatenate_p] = concatenate_backward_rule
update_rules[lax.concatenate_p] = default_update_rule(
  concatenate_backward_rule,
  lambda *args, dimension: lax.concatenate(
      [a for a in args if a is not None], dimension))

def slice_backward_rule(outbox, operand, start_indices, limit_indices, strides):
  if strides is not None:
    raise NotImplementedError('Strided slice is not yet implemented')
  out_start, out_shape = outbox
  in_start = np.add(out_start, start_indices)
  return [in_start], [np.ones(out_shape, int)]

backward_rules[lax.slice_p] = slice_backward_rule
update_rules[lax.slice_p] = default_update_rule(slice_backward_rule,
                                                lambda x, **_: x)

def transpose_backward_rule(outbox, operand, permutation):
  out_start, out_shape = outbox
  inverse_perm = np.argsort(permutation)
  return ([np.take(out_start, inverse_perm)],
          [np.ones(np.take(out_shape, inverse_perm), int)])

def_standard_op(lax.transpose_p, transpose_backward_rule)

def rev_backward_rule(outbox, operand, dimensions):
  out_start, out_shape = outbox
  in_start = [op_size - (start + size) if d in dimensions else start
              for d, (op_size, size, start)
              in enumerate(zip(operand.shape, out_shape, out_start))]
  return [in_start], [np.ones(out_shape, int)]

def_standard_op(lax.rev_p, rev_backward_rule)

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
  return [in_start], [np.ones(in_shape, int) * factor]

def broadcast_in_dim_update_rule(cache, outbox, operand, shape,
                                 broadcast_dimensions):
  out_start, out_shape = outbox
  (in_start,), (in_count,) = broadcast_in_dim_backward_rule(
      outbox, operand, shape, broadcast_dimensions)
  in_part = lax.dynamic_slice(operand, in_start, in_count.shape)
  out_part = lax.broadcast_in_dim(
      in_part, shape=out_shape, broadcast_dimensions=broadcast_dimensions)
  return lax.dynamic_update_slice(cache, out_part, out_start)

# def_standard_op(lax.broadcast_in_dim_p, broadcast_in_dim_backward_rule)
backward_rules[lax.broadcast_in_dim_p] = broadcast_in_dim_backward_rule
update_rules[lax.broadcast_in_dim_p] = broadcast_in_dim_update_rule

def dot_general_backward_rule(outbox, lhs, rhs, dimension_numbers, precision):
  out_start, out_shape = outbox
  outslices = list(zip(*outbox))
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  lhs_other_out_dims = list(range(len(lhs_batch), len(lhs.shape) - len(lhs_contracting)))
  rhs_other_out_dims = list(range(len(rhs_batch) + len(lhs_other_out_dims), len(out_shape)))
  lhs_outbox = unzip2([outslices[d] for d in list(lhs_batch) + lhs_other_out_dims])
  (lhs_start,), (lhs_incount,) = reduce_backward_rule(lhs_outbox, lhs, axes=lhs_contracting)
  lhs_incount *= prod([out_shape[d] for d in rhs_other_out_dims])
  rhs_outbox = unzip2([outslices[d] for d in list(rhs_batch) + rhs_other_out_dims])
  (rhs_start,), (rhs_incount,) = reduce_backward_rule(rhs_outbox, rhs, axes=rhs_contracting)
  rhs_incount *= prod([out_shape[d] for d in lhs_other_out_dims])
  return [lhs_start, rhs_start], [lhs_incount, rhs_incount]

def_standard_op(lax.dot_general_p, dot_general_backward_rule)

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
  padcount = prod(out_shape) - prod(inshape)
  return [instart, ()], [np.ones(inshape), padcount]

def pad_update_rule(cache, outbox, operand, padding_value, padding_config):
  out_start, out_shape = outbox
  (instart, _), (incount, _) = pad_backward_rule(outbox, operand, padding_value, padding_config)
  inslice = lax.dynamic_slice(operand, instart, incount.shape)
  lo, _, interior = unzip3(padding_config)
  dilation = np.array(interior) + 1
  inclip = lambda indices: np.clip(indices, 0, operand.shape)
  next_instart = inclip(lax.lax._ceil_divide(out_start - lo, dilation))
  next_outstart = next_instart * dilation + lo
  _start = next_outstart - out_start
  _stop = _start + jnp.array(incount.shape) * dilation
  out_index = [slice(start, stop, step)
               for start, stop, step in zip(_start, _stop, dilation)]
  out_slice = jnp.full(out_shape, padding_value)
  out_slice = ops.index_update(out_slice, out_index, inslice)
  return lax.dynamic_update_slice(cache, out_slice, out_start)

backward_rules[lax.pad_p] = pad_backward_rule
update_rules[lax.pad_p] = pad_update_rule
