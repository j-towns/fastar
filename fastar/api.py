from jax import jit
from jax import tree

from fastar import core


def as_scan(f, xs):
    traced = jit(f).trace(xs)
    body_fn_flat, init_carry = core.make_scan(traced.jaxpr)
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
