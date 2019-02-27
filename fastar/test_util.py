from jax.util import safe_map
from jax.util import safe_zip
from fastar import accelerate, Parray
import fastar.util as util
import numpy as np

map = safe_map
zip = safe_zip

def increasing_masks(rng, *arrs_raw):
    idxs = shuffled_idxs(rng, *arrs_raw)
    masks = [util.false_mask(arr) for arr in arrs_raw]
    arrs = []
    for argnum, idx in idxs:
        mask = np.copy(masks[argnum])
        mask[idx] = True
        masks[argnum] = mask
        arrs.append(tuple(Parray((arr * mask, mask))
                          for arr, mask in zip(arrs_raw, masks)))
    return arrs

def shuffled_idxs(rng, *arrs):
    idxs = sum(([(argnum, idx) for idx in np.ndindex(np.shape(arr))]
               for argnum, arr in enumerate(arrs)), [])
    perm = rng.permutation(len(idxs))
    return [idxs[i] for i in perm]

def is_subset(mask_1, mask_2):
    return np.all(~mask_1 | mask_2)

def check_ans(ans_old, ans):
    ans_old, mask_old = ans_old
    ans, mask = ans
    assert isinstance(mask, (bool, np.bool_, np.ndarray))
    assert is_subset(mask_old, mask)
    assert is_subset(np.bool_(ans), mask)
    assert np.all(np.where(mask_old, ans == ans_old, True))


def check_custom_input(fun, inputs_from_rng, rtol=1e-5, atol=1e-8, runs=2):
    rng = np.random.RandomState(0)
    for _ in range(runs):
        args = inputs_from_rng(rng)
        ans = fun(*args)
        fun_ac = accelerate(fun)
        masks = increasing_masks(rng, *args)
        args_ = [Parray((arg, util.false_mask(arg))) for arg in args]
        ans_old, fun_ac = fun_ac(*args_)
        for args in masks:
            ans_, fun_ac = fun_ac(*args)
            check_ans(ans_old, ans_)
            ans_old = ans_
        ans_, mask = ans_
        assert np.all(mask)
        try:
            assert np.allclose(ans, ans_, rtol=rtol, atol=atol)
        except AssertionError:
            np.set_printoptions(threshold=np.nan)
            raise AssertionError(
                'Result incorrect: ' + str(ans) + 'vs. \n' + str(ans_) +
                '\n: Differs at: ' + str(np.isclose(ans, ans_, rtol=rtol, atol=atol)) +
                '\n Difference: ' + str(ans - ans_))
        assert ans.dtype == ans_.dtype

def check(fun, *shapes, **kwargs):
    check_custom_input(fun, lambda rng: tuple(rng.randn(*shape)
                                              for shape in shapes), **kwargs)
