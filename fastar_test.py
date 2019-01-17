import numpy as onp
import fastar as fa


def test_rectangular_mask_to_slice():
    shape = (3, 4, 5, 6)
    a = onp.arange(onp.prod(shape)).reshape(shape)
    slc = (slice(0, 2), slice(None, None), slice(2, 4), slice(5, 6))
    msk = onp.zeros_like(a, dtype=bool)
    msk[slc] = True
    slc_ = fa.rectangular_mask_to_slice(msk)
    onp.testing.assert_equal(a[slc], a[slc_])
    onp.testing.assert_equal(a[msk], onp.ravel(a[slc_]))
