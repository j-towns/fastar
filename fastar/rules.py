from functools import partial
from itertools import repeat
import numpy as np
from jax import numpy as jnp, lax
from jax.extend.core import jaxpr_as_fun
from fastar.util import safe_map, unzip2, safe_zip

from fastar.core import register_scanify_rule, ScanConversionError

map = safe_map
zip = safe_zip

def all_equal(xs):
    if len(xs) == 0:
        return True
    else:
        x, *xs = xs
        return all(y == x for y in xs)

def interleave_scanvars(argnums, consts, xs):
    assert len(argums) == len(xs)
    consts_cntr = 0
    xs_cntr = 0
    res = []
    for argnum in range(len(consts) + len(xs)):
        if argnum in argnums:
            res.append(xs[xs_cntr])
            xs_cntr = xs_cntr + 1
        else:
            res.append(consts[consts_cntr])
            consts_cntr = consts_cntr + 1
    return res

def batch_scanify_rule(op, inscanvars, *in_avals, **bind_params):
    # Used when scanning along a batch dimension of op
    assert not op.multiple_results
    _, scanvar_axes = unzip2(inscanvars)
    assert all_equal(scanvar_axes)
    axis = scanvar_axes[0]
    init = None
    def body_fun(carry, *args):
        assert carry == None
        args = list(args)
        for argnum, axis in inscanvars:
            args[argnum] = jnp.expand_dims(args[argnum], axis)
        ans = op.bind(*args, **bind_params)
        return None, lax.squeeze(ans, [axis])
    return init, body_fun, [(0, axis)], []

nary_ops = [
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
    lax.select_n_p,
    lax.integer_pow_p,
]

for op in nary_ops:
    register_scanify_rule(op, partial(batch_scanify_rule, op))

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
def reduce_scanify_rule(op, inscanvars, xs_aval, axes):
    _, [inscan_axis] = unzip2(inscanvars)
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
    scanvar_argnums, scanvar_axes = unzip2(inscanvars)
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
    consts, carry, xs = (
        xs_avals[:num_consts],
        xs_avals[num_consts:num_consts + num_carry],
        xs_avals[num_consts + num_carry:]
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
    )
    out_to_delete = list(range(num_carry))
    return carry, body_fun, out_scanvars, out_to_delete
register_scanify_rule(lax.scan_p, scan_scanify_rule)

#def broadcast_in_dim_scanify_rule(inscanvars, operand, shape,
#                                  broadcast_dimensions, sharding):
#    _, [inscan_axis] = unzip2(inscanvars)
#    if sharding is not None:
#        raise ScanConversionError(
#            "Sharding in broadcast_in_dim not yet supported."
#        )
#    if (operand.shape[inscan_axis] == 1
#        and shape[broadcast_dimensions[inscan_axis]] > 1):
#        raise ScanConversionError(
#            "Global scan along broadcasting axis is not supported."
#        )
#    return lax.broadcast_in_dim_p.bind(
#        operand, shape=shape, broadcast_dimensions=broadcast_dimensions,
#        sharding=sharding
#    ), [(0, broadcast_dimensions[inscan_axis])]
#register_scanify_rule(lax.broadcast_in_dim_p, broadcast_in_dim_scanify_rule)
#
#def conv_general_dilated_scanify_rule(
#    inscanvars, lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
#    dimension_numbers, feature_group_count, batch_group_count, precision,
#    preferred_element_type
#):
#    inscan_argnums, inscan_axes = unzip2(inscanvars)
#    if lhs.ndim > 3:
#        raise ScanConversionError(
#            "Conv with spatial dimension > 1 not yet supported."
#        )
#    if 1 in inscan_argnums:
#        raise ScanConversionError(
#            "Global scan is not currently supported over rhs of "
#            "conv_general_dilated."
#        )
#    [inscan_axis] = inscan_axes
#    if inscan_axis == dimension_numbers.lhs_spec[0]:
#        if batch_group_count > 1:
#            raise ScanConversionError(
#                "Global scan is not yet supported over conv lhs batch axis "
#                "with batch_group_count > 1"
#            )
#        return lax.conv_general_dilated_p.bind(
#            lhs, rhs, window_strides=window_strides, padding=padding,
#            lhs_dilation=lhs_dilation, rhs_dilation=rhs_dilation,
#            dimension_numbers=dimension_numbers,
#            feature_group_count=feature_group_count,
#            batch_group_count=batch_group_count, precision=precision,
#            preferred_element_type=preferred_element_type
#        ), [(0, dimension_numbers.out_spec[0])]
#    if inscan_axis == dimension_numbers.lhs_spec[1]:
#        raise ScanConversionError(
#            "Global scan along feature dimension of conv lhs is not "
#            "supported."
#        )
#    assert inscan_axis == dimension_numbers.lhs_spec[2]
#    window_strides, = window_strides
#    if window_strides > 1:
#        raise ScanConversionError(
#            "Window strides > 1 not yet supported in conv."
#        )
#    window_size = rhs.shape[dimension_numbers.rhs_spec[2]]
#    if padding != ((window_size - 1, 0),):
#        raise ScanConversionError(
#            "Only causal padding is supported in conv."
#        )
#    if lhs_dilation != (1,):
#        raise ScanConversionError(
#            "lhs_dilation > 1 not yet supported in conv."
#        )
#    if rhs_dilation != (1,):
#        raise ScanConversionError(
#            "rhs_dilation > 1 not yet supported in conv."
#        )
#    from IPython.terminal.debugger import set_trace; set_trace()
#register_scanify_rule(lax.conv_general_dilated_p,
#                      conv_general_dilated_scanify_rule)
