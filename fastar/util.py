from copy import deepcopy as copy
from itertools import product, groupby
import numpy as np

from jax.util import safe_map
from jax.util import safe_zip

map = safe_map
zip = safe_zip

def true_mask(val):
    return np.full_like(val, True, dtype=bool)

def false_mask(val):
    return np.full_like(val, False, dtype=bool)

def to_tree(idxs):
    fsts = set(zip(*idxs)[0])
    if len(idxs[0]) > 1:
        return {fst: to_tree([idx[1:] for idx in idxs if idx[0] == fst])
                for fst in fsts}
    else:
        return fsts

def contains_rectangle(idx_tree, rectangle):
    """
    Return True if rectangle is contained in idx_tree, else false.
    """
    (start, stop), *rectangle = rectangle
    if rectangle:
        return all(n in idx_tree and contains_rectangle(idx_tree[n], rectangle)
                   for n in range(start, stop))
    else:
        assert type(idx_tree) is set
        return all(n in idx_tree for n in range(start, stop))

def remove_rectangle(idx_tree, rectangle):
    (start, stop), *rectangle = rectangle
    if rectangle:
        new_tree = {}
        for root, branch in idx_tree.items():
            if start <= root < stop:
                new_branch = remove_rectangle(branch, rectangle)
                if new_branch:
                    new_tree[root] = new_branch
            else:
                new_tree[root] = copy(branch)
    else:
        assert type(idx_tree) is set
        new_tree = set()
        for leaf in idx_tree:
            if not start <= leaf < stop:
                new_tree = new_tree | {leaf}
    return new_tree

def find_rectangle(idx_tree):
    """
    Greedily find a rectangle in idx_tree, return the rectangle and the
    remainder of the tree.
    """
    if type(idx_tree) is dict:
        idx_tree = copy(idx_tree)
        start, *roots = sorted(idx_tree.keys())
        rect = find_rectangle(idx_tree[start])
        stop = start + 1
        while stop in idx_tree and contains_rectangle(idx_tree[stop], rect):
            stop += 1
        return ((start, stop),) + rect
    else:
        assert type(idx_tree) is set
        start = min(idx_tree)
        stop = start + 1
        while stop in idx_tree:
            stop += 1
        return ((start, stop),)

def mask_to_slices(mask):
    """
    Greedily search for rectangular slices in mask.
    """
    rectangles = []
    idx_tree = to_tree(np.argwhere(mask))
    while idx_tree:
        rect = find_rectangle(idx_tree)
        idx_tree = remove_rectangle(idx_tree, rect)
        rectangles.append(rect)
    return [tuple(slice(s, e) for s, e in rect) for rect in rectangles]

def to_tuple_tree(arr):
    return tuple(map(to_tuple_tree, arr) if np.ndim(arr) > 1 else arr)
