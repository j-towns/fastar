import numpy as np
from jax.util import safe_map, safe_zip
from jax.core import Primitive
from jax.interpreters import ad, xla, batching, numpy_eval
from jax.lax import dynamic_update_slice_p

map = safe_map
zip = safe_zip

inplace_dynamic_update_slice_p = Primitive('inplace_dynamic_update_slice')
inplace_dynamic_update_slice_p.def_impl(dynamic_update_slice_p.impl)
inplace_dynamic_update_slice_p.def_abstract_eval(dynamic_update_slice_p.abstract_eval)
for rules in [xla.translations, ad.primitive_jvps, ad.primitive_transposes,
              batching.primitive_batchers]:
  rules[inplace_dynamic_update_slice_p] = rules[dynamic_update_slice_p]

def _numpy_inplace_dynamic_update_slice(operand, update, *start_indices):
  slices = tuple(map(slice, start_indices, np.add(start_indices, update.shape)))
  operand[slices] = update
  return operand

numpy_eval.np_impl[inplace_dynamic_update_slice_p] = \
  _numpy_inplace_dynamic_update_slice

def inplace_dynamic_update_slice(operand, update, start_indices):
  return inplace_dynamic_update_slice_p.bind(operand, update, *start_indices)
