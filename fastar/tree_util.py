from functools import partial
from fastar.core import parray
from jax import tree_util
from jax.util import unzip2, safe_map


map = safe_map

def tree_split(parrays):
  flat, treedef = tree_util.tree_flatten(parrays)
  arrs, knowns = unzip2(flat)
  return map(partial(tree_util.tree_unflatten, treedef), (arrs, knowns))

tree_parray = partial(tree_util.tree_multimap, parray)
