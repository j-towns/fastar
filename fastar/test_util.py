import numpy as np
from jax.util import safe_map
from jax.util import safe_zip

import fastar.util as util
from fastar import accelerate, Parray

map = safe_map
zip = safe_zip


def decreasingly_masked_args(rng, *arrs_raw):
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
    assert is_subset(mask_old, mask)
    assert is_subset(np.bool_(ans), mask)
    assert np.all(np.where(mask_old, ans == ans_old, True))


def check_custom_input(fun, args_from_rng, rtol=1e-5, runs=5):
    rng = np.random.RandomState(0)
    for _ in range(runs):
        args = args_from_rng(rng)
        true_ans = fun(*args)
        first_fun_ac = accelerate(fun)
        dec_masked_args = decreasingly_masked_args(rng, *args)
        initial_masks = [util.false_mask(arg) for arg in args]

        def ar_checked(args):
            masked_ans_old, fun_ac = first_fun_ac(
                *map(Parray, zip(args, initial_masks)))
            for masked_args in dec_masked_args:
                masked_ans, fun_ac = fun_ac(*map(Parray, masked_args))
                check_ans(masked_ans_old, masked_ans)
                masked_ans_old = masked_ans

            ans, mask = masked_ans
            assert np.all(mask)
            return ans

        ans = ar_checked(args)
        assert np.allclose(true_ans, ans, rtol=rtol)
        assert true_ans.dtype == ans.dtype

        def ar(args):
            ans_old, fun_ac = first_fun_ac(
                *map(Parray, zip(args, initial_masks)))
            for masked_args in dec_masked_args:
                (ans, _), fun_ac = fun_ac(*map(Parray, masked_args))

            return ans

        # TODO replace ar with jit(ar):
        ans = ar(args)
        assert np.allclose(true_ans, ans, rtol=rtol)
        assert true_ans.dtype == ans.dtype


def check(fun, *shapes, **kwargs):
    check_custom_input(fun, lambda rng: tuple(rng.randn(*shape)
                                              for shape in shapes), **kwargs)
