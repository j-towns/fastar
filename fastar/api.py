from jax import tree_util, jit
from jax.extend.core import ClosedJaxpr
from jax import tree

from fastar import core


def as_scan(f, xs):
    traced = jit(f).trace(xs)
    traced.in_tree
    init_carry = core.make_init_carry(traced.jaxpr)
    body_fn_flat = core.make_body(traced.jaxpr)
    def body_fn(carry, xs):
        xs_flat, in_tree = tree.flatten(((xs,), {}))
        assert in_tree == traced.in_tree
        carry, out_flat = body_fn_flat(carry, xs_flat)
        out = tree.unflatten(
            tree.structure(traced.out_info),
            out_flat
        )
        return carry, out
    return body_fn, init_carry
