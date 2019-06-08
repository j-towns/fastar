import numpy as np

from jax.util import safe_map
from jax.util import safe_zip

map = safe_map
zip = safe_zip

def true_mask(val):
    if isinstance(val, tuple):
        return tuple(true_mask(v) for v in val)
    elif np.isscalar(val):
        return True
    else:
        return np.full(np.shape(val), True, dtype=bool)

def false_mask(val):
    if isinstance(val, tuple):
        return tuple(false_mask(v) for v in val)
    elif np.isscalar(val):
        return False
    else:
        return np.full(np.shape(val), False, dtype=bool)

def _to_tree(idxs):
    tree = {}
    for idx in idxs:
        branch = tree
        for i in idx:
            branch = branch.setdefault(i, {})
    return tree

_srange = lambda *args: set(range(*args))

def _contains_rectangle(idx_tree, rectangle):
    """
    Return True if rectangle is contained in idx_tree, else False.
    """
    (start, stop), rectangle = rectangle[0], rectangle[1:]
    return all(n in idx_tree and not rectangle
               or _contains_rectangle(idx_tree[n], rectangle)
               for n in range(start, stop))

def _remove_rectangle(idx_tree, rectangle):
    (start, stop), rectangle = rectangle[0], rectangle[1:]
    new_tree = {}
    for root, branch in idx_tree.items():
        if start <= root < stop:
            if rectangle:
                new_branch = _remove_rectangle(branch, rectangle)
                if new_branch:
                    new_tree[root] = new_branch
        else:
            new_tree[root] = branch
    return new_tree

def _find_rectangle(idx_tree):
    """
    Greedily find a rectangle in idx_tree.
    """
    start = min(idx_tree.keys())
    stop = start + 1
    branch = idx_tree[start]
    if branch:
        rect = _find_rectangle(branch)
        while stop in idx_tree and _contains_rectangle(idx_tree[stop], rect):
            stop += 1
        return ((start, stop),) + rect
    else:
        while stop in idx_tree:
            stop += 1
        return (start, stop),

def mask_to_slices(mask):
    """
    Greedily search for rectangular slices in mask.
    """
    if np.shape(mask) == ():
        return [()] if mask else []

    rectangles = []
    idx_tree = _to_tree(np.argwhere(mask))
    while idx_tree:
        rect = _find_rectangle(idx_tree)
        rectangles.append(rect)
        idx_tree = _remove_rectangle(idx_tree, rect)
    return [tuple(slice(s, e) for s, e in rect) for rect in rectangles]

def to_tuple_tree(arr):
    return tuple(map(to_tuple_tree, arr) if np.ndim(arr) > 1 else arr)

def to_numpy(x):
    if isinstance(x, float):
        return np.float64(x)

    if isinstance(x, bool):
        return np.bool_(x)

    if isinstance(x, int):
        return np.int64(x)

    return x
