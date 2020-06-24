from jax.interpreters import xla
from jax import tree_util
import jax.linear_util as lu
from jax.api_util import flatten_fun_nokwargs
import jax.interpreters.partial_eval as pe

from fastar import core
from fastar.jaxpr_util import fastar_jaxpr, tie_the_knot


def lazy_eval(fun, *args):
    args_flat, in_tree  = tree_util.tree_flatten(args)
    f = lu.wrap_init(fun)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    jaxpr, consts, _, out_avals = fastar_jaxpr(flat_fun, *args_flat)
    outs_flat = core.lazy_eval_jaxpr(jaxpr, consts, *args_flat)
    for out, aval in zip(outs_flat, out_avals):
      assert core.get_aval(out) == aval
    return tree_util.tree_unflatten(out_tree(), outs_flat)

def lazy_eval_fixed_point(fun, *mock_args):
    args_flat, in_tree  = tree_util.tree_flatten(mock_args)
    f = lu.wrap_init(fun)
    flat_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
    jaxpr, consts, _, out_avals = tie_the_knot(
        fastar_jaxpr(flat_fun, *args_flat))
    outs_flat = core.lazy_eval_jaxpr(jaxpr, consts)
    for out, aval in zip(outs_flat, out_avals):
      assert core.get_aval(out) == aval
    return tree_util.tree_unflatten(out_tree(), outs_flat)
