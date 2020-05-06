from functools import partial
import jax.lax as lax
import jax.scipy.special as special
from jax.tree_util import Partial
import numpy as onp
import numpy.testing as np_testing
from absl.testing import parameterized
from jax import numpy as np, jit
from jax import test_util as jtu
from fastar import Parray
from fastar.core import _init_env, _update_env
from fastar.util import false_mask, mask_to_slices
from jax.util import safe_map, safe_zip

jit_ = jit

map = safe_map
zip = safe_zip

def increasing_masks(rng, *arrs_raw):
  idxs = shuffled_idxs(rng, *arrs_raw)
  masks = [false_mask(arr) for arr in arrs_raw]
  arrs = []
  for argnum, idx in idxs:
    mask = onp.copy(masks[argnum])
    mask[idx] = True
    masks[argnum] = mask
    arrs.append(tuple(Parray((arr * mask, mask))
                      for arr, mask in zip(arrs_raw, masks)))
  return arrs


def shuffled_idxs(rng, *arrs):
  idxs = sum(([(argnum, idx) for idx in onp.ndindex(np.shape(arr))]
              for argnum, arr in enumerate(arrs)), [])
  perm = rng.permutation(len(idxs))
  return [idxs[i] for i in perm]


def is_subset(mask_1, mask_2):
  return np.all(~mask_1 | mask_2)


def check_ans(ans_old, ans):
  ans_old, mask_old = ans_old
  ans, mask = ans
  assert isinstance(mask, (bool, onp.bool_, onp.ndarray))
  assert is_subset(mask_old, mask)
  assert is_subset(onp.bool_(ans), mask)
  assert np.all(np.where(mask_old, ans == ans_old, True))

def check(fun, *args, rtol=1e-5, atol=1e-8, runs=2):
  rng = onp.random.RandomState(0)
  for _ in range(runs):
    ans = fun(*args)
    fun_ac = accelerate(fun, jit=True)
    masks = increasing_masks(rng, *args)
    args_ = [Parray((arg, false_mask(arg))) for arg in args]
    ans_old, fun_ac = fun_ac(*args_)
    for masked_args in masks:
      ans_, fun_ac = fun_ac(*masked_args)
      check_ans(ans_old, ans_)
      ans_old = ans_
    ans_, mask = ans_
    assert np.all(mask)
    np_testing.assert_allclose(ans, ans_, rtol=rtol, atol=atol)
    assert ans.dtype == ans_.dtype

def accelerate(fun, jit=True):
  """
  Similar to fastar.accelerate but with optional jit.
  """
  def fast_fun(env, *args):
    ans, env = _update_env(fun, args, env)
    return ans, Partial(fast_fun, env)

  fast_fun = jit_(fast_fun) if jit else fast_fun

  def first_fun(*args):
    ans, env = _init_env(fun, args)
    return ans, Partial(fast_fun, env)

  first_fun_ = jit_(first_fun, static_argnums=0) if jit else first_fun
  return first_fun
