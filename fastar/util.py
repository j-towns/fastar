from jax import linear_util as lu
from jax.tree_util import tree_unflatten, tree_flatten
from jax.util import safe_map, unzip2
from jax.core import Literal, Jaxpr, JaxprEqn, Var
from jax import tree_util
import jax.core as jc
from functools import partial

import numpy as onp
from jax import tree_util


map = safe_map

def true_mask(val):
  return onp.full(onp.shape(val), True, dtype=bool)


def false_mask(val):
  return onp.full(onp.shape(val), False, dtype=bool)


def mask_all(parray):
  _, mask = parray
  return onp.all(mask)


class Hashable(object):
  def __init__(self, val):
    self.val = val

  def __hash__(self):
    # We hash known boolean masks on their id because hashing on value is
    # probably too slow, and id should be fine grained enough to get cache hits
    # in most use cases.
    return id(self.val)

  def __eq__(self, other):
    return id(self.val) == id(other.val)

def _rewrap_parrays(treedef, flat):
  """
  Parrays are registered as pytree nodes so that they can pass transparently
  through into jitted functions. This function re-wraps the values in the
  tree-flattened 'flat' in their corresponding Parrays.
  """
  wrapped_flat = []
  flat = flat[::-1]
  def wrap(treedef):
    if all(tree_util.treedef_is_leaf(child) for child in treedef.children()):
      wrapped_flat.append(treedef.unflatten([flat.pop()]))
    else:
      for child in treedef.children():
        wrap(child)
  wrap(treedef)
  return wrapped_flat

class _DontWrap(object):
  # Sometimes we want to unflatten without wrapping in a Parray
  def __init__(self, arr):
    self.val = arr

def fastar_tree_flatten(tree):
  """
  Same as jax.tree_util.tree_flatten except that Parrays are left unflattened.
  """
  flat_raw, treedef_raw = tree_flatten(tree)
  _, treedef = tree_flatten(
      tree_unflatten(treedef_raw, map(_DontWrap, flat_raw)))
  return _rewrap_parrays(treedef_raw, flat_raw), treedef


# Utils for mapping a boolean index array to a list of slices
def _to_tree(idxs):
  tree = {}
  for idx in idxs:
    branch = tree
    for i in idx:
      branch = branch.setdefault(i, {})
  return tree


def _contains_rectangle(idx_tree, rectangle):
  """
  Return True if rectangle is contained in idx_tree, else False.
  """
  (start, stop), rectangle = rectangle[0], rectangle[1:]
  return all(
    n in idx_tree
    and (not rectangle or _contains_rectangle(idx_tree[n], rectangle))
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
  if onp.shape(mask) == ():
    return [()] if mask else []

  rectangles = []
  idx_tree = _to_tree(onp.argwhere(mask))
  while idx_tree:
    rect = _find_rectangle(idx_tree)
    rectangles.append(rect)
    idx_tree = _remove_rectangle(idx_tree, rect)
  return [tuple(slice(s, e) for s, e in rect) for rect in rectangles]


# Move constants inside jaxpr, i.e. make them into 'literals'
# Need a custom literal class because jax.core.literal only supports scalars
class Literal_(Literal):
  __slots__ = ["val"]

  def __init__(self, val):
    self.val = val

  @property
  def aval(self):
    return raise_to_shaped(get_aval(self.val))

  def __hash__(self):
    return id(self.val)

  def __eq__(self, other):
    return self.val is other.val

  def __repr__(self):
    return '{}'.format(self.val)


def submerge_consts(jaxpr, consts, invals=None):
  """
  Replace constvars with literals in jaxpr and its sub-jaxprs.
  """
  # TODO(j-towns): check that consts are in jax.core.literalable_types
  consts = dict(zip(jaxpr.constvars, consts))
  if invals is not None:
    # We're in a call_jaxpr
    new_jaxpr_invars = []
    for var, val in zip(jaxpr.invars, invals):
      if isinstance(val, Var):
        new_jaxpr_invars.append(var)
      else:
        consts[var] = val
  else:
    new_jaxpr_invars = jaxpr.invars
  new_eqns = []
  for eqn in jaxpr.eqns:
    if all(isinstance(var, Literal) or var in consts for var in eqn.invars):
      # Perform constant folding if all inputs to an eqn are known
      in_vals = [var.val if isinstance(var, Literal) else consts[var]
                 for var in eqn.invars]
      call_jaxpr, params = jc.extract_call_jaxpr(eqn.primitive, eqn.params)
      if call_jaxpr:
        subfuns = [lu.wrap_init(partial(jc.eval_jaxpr, call_jaxpr, ()))]
      else:
        subfuns = []
      ans = eqn.primitive.bind(*(subfuns + in_vals), **params)
      if eqn.primitive.multiple_results:
        for outvar, out in zip(eqn.outvars, ans):
          consts[outvar] = out
      else:
        outvar, = eqn.outvars
        consts[outvar] = ans
    else:
      new_invars = [consts[var] if (isinstance(var, Var) and var in consts)
                    else var for var in eqn.invars]
      new_params = dict(eqn.params)
      if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
        new_params['call_jaxpr'] = submerge_consts(eqn.params['call_jaxpr'], [],
                                                   new_invars)
        new_invars = [var for var in new_invars if isinstance(var, Var)]
      else:
        new_invars = [var if isinstance(var, (Var, Literal)) else Literal_(var)
                      for var in new_invars]
      new_eqns.append(JaxprEqn(invars=new_invars, outvars=eqn.outvars,
                               primitive=eqn.primitive, params=new_params))
  return Jaxpr([], new_jaxpr_invars, jaxpr.outvars, new_eqns)
