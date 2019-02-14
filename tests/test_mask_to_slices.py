import numpy as np

from fastar.util import mask_to_slices


def test_scalar_as_array_false():
    assert mask_to_slices(False) == []


def test_scalar_as_array_true():
    assert mask_to_slices(True) == [()]


def test_scalar_false():
    assert mask_to_slices(np.array(False)) == []


def test_scalar_true():
    assert mask_to_slices(np.array(True)) == [()]


def test_vector():
    assert mask_to_slices(np.array([True, False, False, True, True])) == [
        (slice(0, 1, None),), (slice(3, 5, None),)]


def test_matrix():
    assert mask_to_slices(np.array([
        [True, True, False],
        [True, True, True],
        [True, True, False]])) == [
               (slice(0, 3, None), slice(0, 2, None)),
               (slice(1, 2, None), slice(2, 3, None))]
