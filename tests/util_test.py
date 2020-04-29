import numpy as onp
from fastar.util import mask_to_slices


def test_mask_to_slices():
  assert mask_to_slices(False) == []
  assert mask_to_slices(True) == [()]
  assert mask_to_slices(onp.array(False)) == []
  assert mask_to_slices(onp.array(True)) == [()]
  assert mask_to_slices(onp.array([True, False, False, True, True])) == [
    (slice(0, 1, None),), (slice(3, 5, None),)]
  assert mask_to_slices(onp.array([
    [True, True, False],
    [True, True, True],
    [True, True, False]])) == [
           (slice(0, 3, None), slice(0, 2, None)),
           (slice(1, 2, None), slice(2, 3, None))]
