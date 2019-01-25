import numpy as onp

import jax.core as jc
import jax.lax as lax
import jax.interpreters.xla as xla
import jax.interpreters.ad as ad
from jax.util import partial

import fastar.util as util
# Semantically this primitive is no different to lax.conv_general_dilated. We
# need it so that we can correctly propagate fa.Masked values.
def _pop_msk(fun):
    def fun_(*args, **kwargs):
        kwargs.pop('rhs_mask')
        return fun(*args, **kwargs)
    return fun_

_conv_general_dilated_masked_p = jc.Primitive('conv_general_dilated_masked')
_conv_general_dilated_masked_p.def_impl(
    partial(xla.apply_primitive, _conv_general_dilated_masked_p))
_conv_general_dilated_masked_p.def_abstract_eval(_pop_msk(
    partial(lax.standard_abstract_eval, lax.conv_general_dilated_shape_rule,
            lax.conv_general_dilated_dtype_rule)))
xla.translations[_conv_general_dilated_masked_p] = (
    _pop_msk(lax.conv_general_dilated_translation_rule))
ad.defbilinear(_conv_general_dilated_masked_p,
               _pop_msk(lax.conv_general_dilated_transpose_lhs),
               _pop_msk(lax.conv_general_dilated_transpose_rhs))

def conv_general_dilated_masked(
        lhs, rhs, window_strides, padding, lhs_dilation=None,
        rhs_dilation=None, dimension_numbers=None, rhs_mask=None):
    rhs_mask = util.to_tuple_tree(util.true_mask(rhs)
                                  if rhs_mask is None else rhs_mask)
    if type(dimension_numbers) is not lax.ConvDimensionNumbers:
        dimension_numbers = lax.conv_dimension_numbers(
            lhs.shape, rhs.shape, dimension_numbers)
    if isinstance(padding, str):
        lhs_perm, rhs_perm, _ = dimension_numbers
        padding = lax.padtype_to_pads(
            onp.take(lhs.shape, lhs_perm)[2:],
            onp.take(rhs.shape, rhs_perm)[2:], window_strides, padding)
    if lhs_dilation is None:
        lhs_dilation = (1,) * (lhs.ndim - 2)
    if rhs_dilation is None:
        rhs_dilation = (1,) * (rhs.ndim - 2)
    return _conv_general_dilated_masked_p.bind(
        lhs, rhs, window_strides=tuple(window_strides),
        padding=tuple(padding), lhs_dilation=tuple(lhs_dilation),
        rhs_dilation=tuple(rhs_dilation), dimension_numbers=dimension_numbers,
        rhs_mask=rhs_mask, lhs_shape=lhs.shape, rhs_shape=rhs.shape)
