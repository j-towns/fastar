from functools import partial
import jax.lax as lax
import numpy as np
import numpy.testing as np_testing
import jax.numpy as jnp
import jax.test_util as jtu
from jax import vjp
from jax.util import safe_map, safe_zip

from fastar.core import backward_rules
from fastar.box_util import box_to_slice

map = safe_map
zip = safe_zip


def check_backward_rule(primitive, outbox, *invals, **params):
  """
  Checks the backward rule for a lax op against the vjp.
  """
  in_starts, in_counts = backward_rules[primitive](outbox, *invals, **params)
  cotangents = tuple(np.zeros_like(val) for val in invals)
  for ct, start, count in zip(cotangents, in_starts, in_counts):
    ct[box_to_slice((start, count.shape))] += count

  primitive_bind = partial(primitive.bind, **params)
  out, vjp_ = vjp(primitive_bind, *invals)
  out_ct = np.zeros_like(out)
  out_ct[box_to_slice(outbox)] = 1
  cotangents_expected = vjp_(out_ct)
  jtu.check_eq(cotangents, cotangents_expected)
