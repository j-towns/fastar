import jax.core as jc
from jax.interpreters import xla
import jax.test_util as jtu
from jax import jit, make_jaxpr
import numpy as jnp

import fastar.jaxpr_util as ju


def test_inline_calls():
  @jit
  def f(x):
    x = 2 * x + 1
    y = x - jnp.array([3, 4])

    @jit
    def g(x):
      return 2 * x + y
    return g(x)

  x = jnp.array([1., 2.])
  jaxpr = make_jaxpr(f)(x)
  jaxpr, consts = jaxpr.jaxpr, jaxpr.consts
  inlined_jaxpr = ju.inline_calls(jaxpr)
  jc.check_jaxpr(inlined_jaxpr)
  for e in inlined_jaxpr.eqns:
    assert not (e.primitive is xla.xla_call_p)
  jtu.check_close([f(x)], jc.eval_jaxpr(inlined_jaxpr, consts, x))
