from jax.util import safe_map
from jax.util import safe_zip
from itertools import product
from fastar import accelerate, Parray
import fastar.util as util
import numpy as np
import numpy.random

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
        arrs.append(tuple(Parray((arr, mask))
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
    assert is_subset(mask_old, mask)
    assert is_subset(np.bool_(ans), mask)
    assert np.all(np.where(mask_old, ans == ans_old, True))

def check_fun(rng, fun, *args):
    ans = fun(*args)
    fun_ac = accelerate(fun)
    args_ = [Parray((arg, util.false_mask(arg))) for arg in args]
    ans_old, fun_ac = fun_ac(*args_)
    for args in increasing_masks(rng, *args):
        ans_, fun_ac = fun_ac(*args)
        check_ans(ans_old, ans_)
        ans_old = ans_
    ans_, mask = ans_
    assert np.all(mask)
    assert np.all(ans == ans_)
