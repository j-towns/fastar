from functools import partial
from itertools import repeat
import numpy as np
from jax import numpy as jnp, lax
from jax.extend.core import jaxpr_as_fun
import jax
from fastar.util import safe_map, unzip2, unzip3, safe_zip

from fastar.core import register_scanify_rule, ScanConversionError

map = safe_map
zip = safe_zip


def all_equal(xs):
    if len(xs) == 0:
        return True
    else:
        x, *xs = xs
        return all(y == x for y in xs)

def batch_scanify_rule(op, inscanvars, *in_avals, **bind_params):
    # Used when scanning along a batch dimension of op
    assert not op.multiple_results
    _, scanvar_axes, strides = unzip3(inscanvars)
    assert all_equal(scanvar_axes)
    assert all_equal(strides)
    axis, stride = scanvar_axes[0], strides[0]
    if stride == 1:
        carry_init = None

        def body_fn(carry, *args):
            assert carry is None
            args = list(args)
            for argnum, axis, _ in inscanvars:
                args[argnum] = jnp.expand_dims(args[argnum], axis)
            ans = op.bind(*args, **bind_params)
            return None, lax.squeeze(ans, [axis])
    else:
        carry_init = 0

        def body_fn(i, *args):
            args = list(args)
            for argnum, axis, _ in inscanvars:
                args[argnum] = jnp.expand_dims(args[argnum], axis)
            ans = lax.squeeze(op.bind(*args, **bind_params), [axis])
            return i + 1, lax.cond(
                i % stride,
                lambda : jnp.zeros_like(ans),
                lambda : ans,
            )
    return carry_init, body_fn, [(0, axis, stride)], []

nary_ops = [
    lax.abs_p,
    lax.acos_p,
    lax.acosh_p,
    lax.add_p,
    lax.and_p,
    lax.asin_p,
    lax.asinh_p,
    lax.atan_p,
    lax.atan2_p,
    lax.atanh_p,
    lax.bessel_i0e_p,
    lax.bessel_i1e_p,
    lax.cbrt_p,
    lax.ceil_p,
    lax.clz_p,
    lax.complex_p,
    lax.conj_p,
    lax.convert_element_type_p,
    lax.cos_p,
    lax.cosh_p,
    lax.digamma_p,
    lax.div_p,
    lax.eq_p,
    lax.erf_inv_p,
    lax.erf_p,
    lax.erfc_p,
    lax.exp_p,
    lax.exp2_p,
    lax.expm1_p,
    lax.floor_p,
    lax.ge_p,
    lax.gt_p,
    lax.igamma_grad_a_p,
    lax.igamma_p,
    lax.igammac_p,
    lax.imag_p,
    lax.integer_pow_p,
    lax.is_finite_p,
    lax.le_p,
    lax.lgamma_p,
    lax.log_p,
    lax.log1p_p,
    lax.logistic_p,
    lax.lt_p,
    lax.max_p,
    lax.min_p,
    lax.mul_p,
    lax.ne_p,
    lax.neg_p,
    lax.nextafter_p,
    lax.not_p,
    lax.or_p,
    lax.polygamma_p,
    lax.population_count_p,
    lax.pow_p,
    lax.real_p,
    lax.regularized_incomplete_beta_p,
    lax.rem_p,
    lax.round_p,
    lax.rsqrt_p,
    lax.select_n_p,
    lax.shift_left_p,
    lax.shift_right_arithmetic_p,
    lax.shift_right_logical_p,
    lax.sign_p,
    lax.sin_p,
    lax.sinh_p,
    lax.sqrt_p,
    lax.square_p,
    lax.sub_p,
    lax.tan_p,
    lax.tanh_p,
    lax.xor_p,
    lax.zeta_p,
]

def nary_op_scanify_rule(op, inscanvars, *avals, **kwargs):
    argnums, axes, strides = unzip3(inscanvars)
    axis = axes[0]
    stride = strides[0]
    if not all(a == axis for a in axes[1:]):
        # TODO: more detail
        raise ScanConversionError(
            "All scanned inputs to nary op must be scanned along same axis"
        )
    if not all(s == stride for s in strides):
        raise ScanConversionError(
            "All scanned inputs to nary op must have same stride/scale"
        )
    if (any(avals[n].shape[axis] == 1 for n in argnums)
            and any(a.shape[axis] > 1 for a in avals)):
        # TODO: more detail
        raise ScanConversionError(
            "Broadcasting scanned variable along scanned axis is not "
            "supported"
        )
    if all(a.ndim == 0 or a.shape[axis] == 1
           for i, a in enumerate(avals) if i not in argnums):
        return batch_scanify_rule(op, inscanvars, *avals, **kwargs)
    init = 0
    def body_fn(counter, *args):
        args = [
            a if i in argnums else lax.dynamic_index_in_dim(
                a, counter, axis, False
            ) for i, a in enumerate(args)
        ]
        ans = op.bind(*args, **kwargs)
        if stride > 1:
            ans = lax.cond(
                counter % stride,
                jnp.zeros_like(ans),
                ans
            )
        return counter + 1, ans
    return init, body_fn, [(0, axis, stride)], []

for op in nary_ops:
    register_scanify_rule(op, partial(nary_op_scanify_rule, op))

reduce_ops = [
    lax.reduce_sum_p,
    lax.reduce_prod_p,
    lax.reduce_max_p,
    lax.reduce_min_p,
    lax.reduce_or_p,
    lax.reduce_and_p,
    lax.reduce_xor_p,
    lax.argmax_p,
    lax.argmin_p,
]
def reduce_scanify_rule(op, inscanvars, xs_aval, axes):
    _, [inscan_axis], _ = unzip3(inscanvars)
    if inscan_axis in set(axes):
        raise ScanConversionError(
            "Global scan operating along reduce axis is not supported."
        )
    return batch_scanify_rule(
        op, inscanvars, xs_aval, axes=axes
    )
for op in reduce_ops:
    register_scanify_rule(op, partial(reduce_scanify_rule, op))

def scan_scanify_rule(inscanvars, *xs_avals, _split_transpose, jaxpr, length,
                      linear, num_carry, num_consts, reverse, unroll):
    scanvar_argnums, scanvar_axes, scanvar_strides = unzip3(inscanvars)
    xs_argnums = set(range(num_consts + num_carry, len(xs_avals)))
    if not set(scanvar_argnums) <= xs_argnums:
        raise ScanConversionError(
            "Global scan along an axis of a constant or carry in a call to "
            "scan. This is not currently supported, but could be in future.")
    if not set(scanvar_argnums) == xs_argnums:
        # TODO: Make this error more specific
        raise ScanConversionError(
            "Global scan over some, but not all, of the inputs of a scan is "
            "not suppoorted"
        )
    if not all(a == 0 for a in scanvar_axes):
        # TODO: Make this error more specific
        raise ScanConversionError(
            "Mismatch between global scan axis and scan axis"
        )
    if not all(s == 1 for s in scanvar_strides):
        raise ScanConversionError(
            "Scan with strided input not yet implemented"
        )
    consts, carry = (
        xs_avals[:num_consts],
        xs_avals[num_consts:num_consts + num_carry],
    )
    def body_fun(carry, *args):
        carry_and_x = jaxpr_as_fun(jaxpr)(*(
            consts + carry + args[num_consts + num_carry:]
        ))
        return (
            tuple(carry_and_x[:num_carry]),
            carry_and_x
        )

    out_scanvars = zip(
        range(num_carry, len(jaxpr.out_avals)),
        repeat(0, len(jaxpr.out_avals) - num_carry),
        repeat(1, len(jaxpr.out_avals) - num_carry),
    )
    out_to_delete = list(range(num_carry))
    return carry, body_fun, out_scanvars, out_to_delete
register_scanify_rule(lax.scan_p, scan_scanify_rule)

def broadcast_in_dim_scanify_rule(inscanvars, operand, shape,
                                  broadcast_dimensions, sharding):
    [(_, inscan_axis, _)] = inscanvars
    if sharding is not None:
        raise ScanConversionError(
            "Sharding in broadcast_in_dim not yet supported."
        )
    if (operand.shape[inscan_axis]
            < shape[broadcast_dimensions[inscan_axis]]):
        raise ScanConversionError(
            "Global scan along broadcasting axis is not supported."
        )
    shape = list(shape)
    shape[broadcast_dimensions[inscan_axis]] = 1
    shape = tuple(shape)
    return batch_scanify_rule(
        lax.broadcast_in_dim_p, inscanvars, operand, shape=shape,
        broadcast_dimensions=broadcast_dimensions, sharding=sharding
    )
register_scanify_rule(lax.broadcast_in_dim_p, broadcast_in_dim_scanify_rule)

def _perm_inverse(p):
    p = np.asarray(p)
    s = np.empty_like(p)
    s[p] = np.arange(len(p))
    return s

def transpose_scanify_rule(inscanvars, operand, permutation):
    [(argnum, in_axis, in_stride)] = inscanvars
    assert argnum == 0
    out_axis = _perm_inverse(permutation)[in_axis]
    def body_fn(carry, x):
        return None, lax.squeeze(
            lax.transpose(
                jnp.expand_dims(x, in_axis), permutation
            ), [out_axis]
        )
    return None, body_fn, [(0, out_axis, in_stride)], []
register_scanify_rule(lax.transpose_p, transpose_scanify_rule)

def conv_general_dilated_scanify_rule(
    inscanvars, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
    dimension_numbers, feature_group_count, batch_group_count, precision,
    preferred_element_type
):
    inscan_argnums, inscan_axes, inscan_strides = unzip3(inscanvars)
    if 1 in inscan_argnums:
        raise ScanConversionError(
            "Global scan is not currently supported over rhs of "
            "conv_general_dilated."
        )
    [inscan_axis] = inscan_axes
    if inscan_axis == dimension_numbers.lhs_spec[0]:
        if batch_group_count > 1:
            raise ScanConversionError(
                "Global scan is not yet supported over conv lhs batch axis "
                "with batch_group_count > 1."
            )
        return batch_scanify_rule(
            lax.conv_general_dilated_p, inscanvars, lhs, rhs,
            window_strides=window_strides, padding=padding,
            lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
            dimension_numbers=dimension_numbers,
            feature_group_count=feature_group_count,
            batch_group_count=batch_group_count, precision=precision,
            preferred_element_type=preferred_element_type
        )
    [inscan_stride] = inscan_strides
    if lhs.ndim > 3:
        raise ScanConversionError(
            "Converting conv with spatial dimension > 1 not yet supported."
        )
    if inscan_axis == dimension_numbers.lhs_spec[1]:
        raise ScanConversionError(
            "Global scan along feature dimension of conv lhs is not "
            "supported."
        )
    assert inscan_axis == dimension_numbers.lhs_spec[2]
    outscan_axis = dimension_numbers.out_spec[2]
    window_stride, = window_strides
    rhs_dilation, = rhs_dilation
    if window_stride > 1:
        raise ScanConversionError(
            "Window strides > 1 not yet supported in conv."
        )
    window_size = rhs.shape[dimension_numbers.rhs_spec[2]]
    if padding != ((rhs_dilation * (window_size - 1), 0),):
        raise ScanConversionError(
            "Only causal padding is supported in conv."
        )
    if lhs_dilation != (1,):
        raise ScanConversionError(
            "lhs_dilation > 1 not yet supported in conv."
        )
    carry_shape = list(lhs.shape)
    carry_shape[inscan_axis] = rhs_dilation * (window_size - 1)
    if inscan_stride == 1:
        carry_init = jnp.zeros(carry_shape, lhs.dtype)
        def body_fn(carry, x, rhs):
            lhs = lax.concatenate(
                [carry, jnp.expand_dims(x, inscan_axis)],
                inscan_axis
            )
            # TODO: Consider using a dot instead of conv
            out = lax.conv_general_dilated(
                lhs, rhs, window_strides=window_strides, padding="VALID",
                lhs_dilation=lhs_dilation, rhs_dilation=(rhs_dilation,),
                dimension_numbers=dimension_numbers,
                feature_group_count=feature_group_count,
                batch_group_count=batch_group_count, precision=precision,
                preferred_element_type=preferred_element_type,
            )
            out = lax.squeeze(out, [outscan_axis])
            carry_new = lax.slice_in_dim(
                lhs, 1, lhs.shape[inscan_axis], 1, inscan_axis
            )
            return carry_new, out
    else:
        carry_init = 0, jnp.zeros(carry_shape, lhs.dtype)
        def body_fn(i_and_carry, x, rhs):
            i, carry = i_and_carry
            lhs = lax.concatenate(
                [carry, jnp.expand_dims(x, inscan_axis)],
                inscan_axis
            )
            out = lax.conv_general_dilated(
                lhs, rhs, window_strides=window_strides, padding="VALID",
                lhs_dilation=lhs_dilation, rhs_dilation=(rhs_dilation,),
                dimension_numbers=dimension_numbers,
                feature_group_count=feature_group_count,
                batch_group_count=batch_group_count, precision=precision,
                preferred_element_type=preferred_element_type,
            )
            out = lax.squeeze(out, [outscan_axis])
            carry_new = lax.slice_in_dim(
                lhs, 1, lhs.shape[inscan_axis], 1, inscan_axis
            )
            return lax.cond(
                i % inscan_stride,
                lambda : ((i + 1, carry), jnp.zeros_like(out)),
                lambda : ((i + 1, carry_new), out),
            )
    return carry_init, body_fn, [(0, outscan_axis, inscan_stride)], []

register_scanify_rule(
    lax.conv_general_dilated_p, conv_general_dilated_scanify_rule
)

def slice_scanify_rule(
   inscanvars, operand, start_indices, limit_indices, strides
):
    [(_, in_axis, in_stride)] = inscanvars
    if (limit_indices[in_axis] - start_indices[in_axis]
            < operand.shape[in_axis]):
        raise ScanConversionError(
            "Slice must be over the full scanned axis"
        )
    if strides is not None and operand.shape[in_axis] % strides[in_axis]:
        raise ScanConversionError(
            "Strided slice along scan axis must have a stride which exactly "
            "exactly divides the input axis size"
        )

    start_indices_ = list(start_indices)
    start_indices_.pop(in_axis)
    limit_indices_ = list(limit_indices)
    limit_indices_.pop(in_axis)
    if strides is not None:
        strides_ = list(strides)
        strides_.pop(in_axis)
    else:
        strides_ = None

    def body_fn(carry, operand):
        assert carry is None
        return carry, lax.slice(
            operand, start_indices_, limit_indices_, strides_
        )

    out_stride = in_stride * (strides[in_axis] if strides is not None else 1)
    return None, body_fn, [(0, in_axis, in_stride * out_stride)], []
register_scanify_rule(lax.slice_p, slice_scanify_rule)

def pad_scanify_rule(
    inscanvars, operand, padding_value, padding_config
):
    assert len(inscanvars) == 1
    [(argnum, axis, in_stride)] = inscanvars
    assert argnum == 0  # Shouldn't be possible to scan over scalar
                           # padding_value
    scan_pad_start, scan_pad_end, scan_pad_interior = padding_config[axis]
    if not scan_pad_start == 0:
        raise ScanConversionError(
            "Padding at the beginning of a scanned axis is not yet "
            "supported"
        )
    if not scan_pad_end == scan_pad_interior:
        raise ScanConversionError(
            "End padding on scanned axis must be equal to interior padding"
        )
    dilation = scan_pad_interior + 1
    if in_stride % dilation:
        raise ScanConversionError(
            "Pad dilation must exactly divide the input stride along scanned "
            "axis"
        )
    out_stride = in_stride // dilation
    padding_config_ = list(padding_config)
    padding_config_.pop(axis)
    def body_fn(i, operand, padding_value):
        ans = lax.pad(operand, padding_value, padding_config_)
        return i + 1, lax.cond(
            i % in_stride,
            lambda : jnp.full_like(ans, padding_value),
            lambda : ans,
        )
    return 0, body_fn, [(0, axis, out_stride)], []
register_scanify_rule(lax.pad_p, pad_scanify_rule)
