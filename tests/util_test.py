import numpy as onp
from jax import jit
from jax.tree_util import tree_flatten
from jax import numpy as np
from fastar import util
from fastar import core


def test_mask_to_slices():
  assert util.mask_to_slices(False) == []
  assert util.mask_to_slices(True) == [()]
  assert util.mask_to_slices(onp.array(False)) == []
  assert util.mask_to_slices(onp.array(True)) == [()]
  assert util.mask_to_slices(onp.array([True, False, False, True, True])) == [
    (slice(0, 1, None),), (slice(3, 5, None),)]
  assert util.mask_to_slices(onp.array([
    [True, True, False],
    [True, True, True],
    [True, True, False]])) == [
           (slice(0, 3, None), slice(0, 2, None)),
           (slice(1, 2, None), slice(2, 3, None))]

def test_submerge_consts():
  @jit
  def f(x):
    return x ** np.array([2., 3.])

  def g(x):
    y = x + 2.
    y = f(y)
    return y * np.array([1., 3.])


  args = (core.parray(np.array([1., 2.]), onp.array([True, True])),)
  args_flat, in_tree = util.fastar_tree_flatten(args)
  avals = tuple(core._get_aval(arg) for arg, _ in args_flat)
  jaxpr, consts, out_tree = core._fastar_jaxpr(g, in_tree, avals)
  assert repr(jaxpr) == """{ lambda  ; a.
  let c = add a 2.0
      d = xla_call[ backend=None
                    call_jaxpr={ lambda  ; a.
                                 let c = pow a [2. 3.]
                                 in (c,) }
                    device=None
                    name=f ] c
      f = mul d [1. 3.]
  in (f,) }"""
