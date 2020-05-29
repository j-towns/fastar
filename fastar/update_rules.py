from functools import reduce
from operator import and_
from warnings import warn

from jax import lax
from jax.util import safe_map, safe_zip
import jax.numpy as np
from jax.scipy import special
from jax.util import curry
from jax.interpreters import xla
from jax.ops import index_update
from jax.core import Tracer

import numpy as onp

from fastar.core import update_rules, Parray
from fastar.util import mask_to_slices

map = safe_map
zip = safe_zip

def _unbroadcast_slice(s, shape):
  return tuple(s if dim_sz > 1 else slice(None)
               for s, dim_sz in zip(s, shape)) if shape else ()

def _slice(arr, slc):
  # We use lax.slice when possible because it simplifies jaxprs and seems to be
  # a bit faster to compile/run than gather, at least on CPU.
  assert type(slc) is tuple and len(slc) == arr.ndim
  starts = []
  stops  = []
  steps  = []
  for s, dim_sz in zip(slc, arr.shape):
    starts.append(0      if s.start is None else s.start)
    stops. append(dim_sz if s.stop  is None else s.stop )
    steps. append(1      if s.step  is None else s.step )
  return lax.slice(arr, starts, stops, steps)

def _update_slice(arr, slc, new):
  # We use lax.dynamic_update_slice when possible because it simplifies jaxprs
  # and seems to be a bit faster to compile/run than index_update, at least on
  # CPU.
  assert type(slc) is tuple and len(slc) == arr.ndim
  starts = []
  stops  = []
  steps  = []
  for s, dim_sz in zip(slc, arr.shape):
    starts.append(0      if s.start is None else s.start)
    stops. append(dim_sz if s.stop  is None else s.stop )
    assert s.step is None or s.step == 1
  assert new.shape == tuple(onp.subtract(stops, starts))
  return lax.dynamic_update_slice(arr, new, starts)

# n-ary elementwise operators with broadcasting
@curry
def _naryop_update(op, ans, *args):
  args, arg_masks = zip(*args)
  args = map(np.asarray, args)
  ans, ans_mask = ans
  new_ans_mask = reduce(and_, arg_masks)
  slices = mask_to_slices(new_ans_mask & ~ ans_mask)
  for s in slices:
    part_args = [_slice(a, _unbroadcast_slice(s, onp.shape(a))) for a in args]
    ans = _update_slice(ans, s, op.bind(*part_args))
  return Parray((ans, new_ans_mask))


_naryops = [
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

for op in _naryops:
  update_rules[op] = _naryop_update(op)


# Cheap ops, these are assumed to require little or no computation
@curry
def _cheap_op_update(op, old_out, *args, **params):
  args, args_mask = zip(*args)
  # TODO: use onp equivalents to process the masks
  return Parray((op.bind(*args, **params),
                 onp.bool_(op.bind(*args_mask, **params))))


_cheap_ops = [
  lax.broadcast_p,
  lax.broadcast_in_dim_p,
  lax.concatenate_p,
  lax.convert_element_type_p,
  lax.reshape_p,
  lax.rev_p,
  lax.slice_p,
  lax.tie_in_p,
  lax.transpose_p,
]

for op in _cheap_ops:
  update_rules[op] = _cheap_op_update(op)


def _gather_update(
    old_out, operand, start_indices, dimension_numbers, slice_sizes,
    **params):
  # Treat gather as a cheap op, but need to handle start_indices correctly
  operand, operand_mask = operand
  start_indices, start_indices_mask = start_indices
  assert onp.all(start_indices_mask)
  # Treat gather as a cheap op
  return Parray((
    lax.gather_p.bind(
      operand, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, **params),
    onp.asarray(lax.gather_p.bind(
      operand_mask, start_indices, dimension_numbers=dimension_numbers,
      slice_sizes=slice_sizes, **params))))


update_rules[lax.gather_p] = _gather_update


# Reductions
@curry
def _reduce_update(op, ans, a, **params):
  a, a_mask = a
  ans, ans_mask = ans
  axes = params['axes']
  new_ans_mask = onp.all(a_mask, axes)
  slices = mask_to_slices(new_ans_mask & ~ ans_mask)
  for s in slices:
    a_slice = list(s)
    for axis in axes:
      a_slice.insert(axis, slice(None))
    a_part = _slice(a, tuple(a_slice))
    if 'input_shape' in params:
      params['input_shape'] = onp.shape(a_part)
    ans = _update_slice(ans, s, op.bind(a_part, **params))
  return Parray((ans, new_ans_mask))


reduce_ops = [
  lax.reduce_and_p ,
  lax.reduce_max_p,
  lax.reduce_min_p,
  lax.reduce_or_p ,
  lax.reduce_prod_p,
  lax.reduce_sum_p,
]

for op in reduce_ops:
  update_rules[op] = _reduce_update(op)


def _dot_general_update(ans, a, b, dimension_numbers, precision=None):
  a, a_mask = a
  b, b_mask = b
  ansval, ansmask = ans
  (a_cont_dims, b_cont_dims), (a_btch_dims, b_btch_dims) = dimension_numbers

  a_outr_dims = tuple(d for d in range(a.ndim)
                      if d not in a_cont_dims + a_btch_dims)
  b_outr_dims = tuple(d for d in range(b.ndim)
                      if d not in b_cont_dims + b_btch_dims)

  cont_ndim = len(a_cont_dims)
  btch_ndim = len(a_btch_dims)

  # Batch dims to front
  a_mask = onp.moveaxis(a_mask, a_btch_dims, range(btch_ndim))
  a_cont_dims_ = tuple(c + sum(1 for b in a_btch_dims if b > c)
                       for c in a_cont_dims)
  b_mask = onp.moveaxis(b_mask, b_btch_dims, range(len(b_btch_dims)))
  b_cont_dims_ = tuple(c + sum(1 for b in b_btch_dims if b > c)
                       for c in b_cont_dims)

  a_mask = onp.all(a_mask, a_cont_dims_)
  b_mask = onp.all(b_mask, b_cont_dims_)

  new_ansmask = (
      a_mask[(Ellipsis,) + len(b_outr_dims) * (onp.newaxis,)]
      & b_mask[btch_ndim * (slice(None),)
               + len(a_outr_dims) * (onp.newaxis,)])
  for s in mask_to_slices(new_ansmask & ~ ansmask):
    s_btch, s_a, s_b = (s[:btch_ndim],
                        s[btch_ndim:btch_ndim + len(a_outr_dims)],
                        s[btch_ndim + len(a_outr_dims):])
    a_slice = tuple((s_btch + s_a + cont_ndim * (slice(None),))[d] for d in
                    onp.argsort(a_btch_dims + a_outr_dims + a_cont_dims))
    b_slice = tuple((s_btch + s_b + cont_ndim * (slice(None),))[d] for d in
                    onp.argsort(b_btch_dims + b_outr_dims + b_cont_dims))
    ansval = _update_slice(ansval, s,
                          lax.dot_general_p.bind(
                            _slice(a, a_slice), _slice(b, b_slice),
                            dimension_numbers=dimension_numbers,
                            precision=precision))
  return Parray((ansval, new_ansmask))


update_rules[lax.dot_general_p] = _dot_general_update


def _pad_update(old_out, input, padding_value, padding_config):
  for (lo, hi, interior) in padding_config:
    if interior > 0 and (lo < 0 or hi < 0): raise NotImplementedError(
      "Interior and negative padding on same axis not yet implemented.")

  input, input_mask = input
  padding_value, padding_value_mask = padding_value
  outval, old_outmask = old_out

  unpad_slice = tuple(
    slice(lo if lo > 0 else None,
          -hi if hi > 0 else None,
          None if interior == 0 else interior + 1)
    for (lo, hi, interior) in padding_config)
  unpad = lambda x: x[unpad_slice]

  def pad_slices():
    for index, (lo, hi, interior) in enumerate(padding_config):
      def nonoverlapping(s):
        return unpad_slice[:index] + (s,) + \
               tuple([slice(None)] * (old_outmask.ndim - index - 1))

      if lo > 0:
        yield nonoverlapping(slice(lo))

      for i in range(interior):
        yield nonoverlapping(slice(lo + i + 1, -hi, (interior + 1)))

      if hi > 0:
        yield nonoverlapping(slice(-hi, None))

  old_padding_value_mask = any(onp.any(old_outmask[s]) for s in pad_slices())
  new_padding_value_mask = padding_value_mask and not old_padding_value_mask

  output_mask = old_outmask.copy()

  input_crop_slice = tuple(
    slice(-lo if lo < 0 else None,
          hi if hi < 0 else None)
    for (lo, hi, _) in padding_config)

  cropped_input_mask = input_mask[input_crop_slice]
  output_mask[unpad_slice] = cropped_input_mask
  if new_padding_value_mask:
    for s in pad_slices():
      outval = index_update(outval, s, np.broadcast_to(
        padding_value, output_mask[s].shape))
      output_mask[s] = True

  cropped_input_new_mask = cropped_input_mask & ~unpad(old_outmask)
  cropped_input_slices = mask_to_slices(cropped_input_new_mask)
  for cropped_input_slice in cropped_input_slices:
    output_slice = tuple(
      slice((lo if lo > 0 else 0) + s.start * (interior + 1),
            (lo if lo > 0 else 0) + s.stop * (interior + 1),
            interior + 1)
      for s, (lo, _, interior) in
      zip(cropped_input_slice, padding_config))

    input_slice = tuple(
      slice(s.start + (-lo if lo < 0 else 0),
            s.stop + (-lo if lo < 0 else 0), s.step)
      for s, (lo, _, interior) in
      zip(cropped_input_slice, padding_config))

    # assert np.all(outval[output_slice] == init_value)
    outval = index_update(outval, output_slice, input[input_slice])

  return Parray((outval, output_mask))


update_rules[lax.pad_p] = _pad_update

def _filter_nonzero(arr):
  # We attempt to automatically detect where zeros are in conv filters in order
  # to support masked convolution.
  if isinstance(arr, Tracer):
    warn("Unable to detect locations of zeros in conv filter. If you're using "
         "masked convolution with FastAR you need to ensure that the filter "
         "has no dependence on the input to the accelerated function.")
    nonzero = np.zeros(arr.shape)
  else:
    nonzero = arr != 0
  return nonzero

def _conv_general_dilated_outmask(lhs_mask, rhs_nonzero, **params):
  # Note: we assume that rhs_mask doesn't change
  lhs_unknown, rhs_nonzero = onp.float32(~lhs_mask), np.float32(rhs_nonzero)
  out_unknown = onp.array(
      lax.conv_general_dilated(lhs_unknown, rhs_nonzero, **params))
  return out_unknown == 0


def _conv_general_dilated_update_slice_op(
    slc, out, lhs, rhs, window_strides, padding,
    lhs_dilation=None, rhs_dilation=None, dimension_numbers=None,
    precision=None):
  # TODO: use rhs zero locations to avoid unnecessary computation
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  lhs_shape = onp.take(onp.shape(lhs), lhs_spec)
  rhs_shape = onp.take(onp.shape(rhs), rhs_spec)
  out_slc = [slc[i] for i in out_spec]
  pad_low, _ = onp.transpose(padding)
  window_shape = lax.lax._dilate_shape(rhs_shape, rhs_dilation)[2:]
  out_start, out_stop = onp.transpose([[s.start, s.stop] for s in out_slc])
  out_start_dilated = out_start[2:] * onp.array(window_strides)
  out_stop_dilated = (out_stop[2:] - 1) * onp.array(window_strides) + 1
  lhs_start_dilated = out_start_dilated - pad_low
  lhs_stop_dilated = out_stop_dilated + window_shape - 1 - pad_low
  lhs_start, lhs_stop = onp.clip(
    (onp.array(
      [lhs_start_dilated, lhs_stop_dilated]) - 1) // lhs_dilation + 1, 0,
    lhs_shape[2:])

  if onp.any(lhs_start == lhs_stop):
    return out

  sub_pad_low = onp.maximum(
    onp.multiply(lhs_start, lhs_dilation) - lhs_start_dilated, 0)
  sub_pad_high = onp.maximum(
    lhs_stop_dilated - onp.multiply((lhs_stop - 1), lhs_dilation) - 1, 0)
  sub_padding = zip(sub_pad_low, sub_pad_high)
  lhs_slice = ((out_slc[0], slice(None)) +
               tuple(slice(int(s), int(e)) for s, e in
                     zip(lhs_start, lhs_stop)))
  new = lax.conv_general_dilated(
    _slice(lhs, tuple(onp.take(lhs_slice, onp.argsort(lhs_spec)))), rhs,
    window_strides=window_strides, padding=sub_padding,
    lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
    dimension_numbers=dimension_numbers, precision=precision)
  return _update_slice(out, slc, new)


def _conv_general_dilated_update(
    old_out, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, feature_group_count, batch_group_count, lhs_shape,
    rhs_shape, precision):
  lhs, lhs_mask = lhs
  rhs, rhs_mask = rhs

  if not onp.all(rhs_mask) or feature_group_count > 1 or batch_group_count > 1:
    raise NotImplementedError

  outval, old_outmask = old_out

  outmask = _conv_general_dilated_outmask(
    lhs_mask, _filter_nonzero(rhs), window_strides=window_strides,
    padding=padding, lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
    dimension_numbers=dimension_numbers)

  new_mask = outmask & ~old_outmask
  for slice in mask_to_slices(new_mask):
    outval = _conv_general_dilated_update_slice_op(
      slice, outval, lhs, rhs, window_strides, padding,
      lhs_dilation, rhs_dilation, dimension_numbers, precision)

  return Parray((outval, outmask))


update_rules[lax.conv_general_dilated_p] = _conv_general_dilated_update
