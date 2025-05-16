# Notes
# =====
# Had a new idea today (30/4/2025) to have a first pass which converts all
# sequence-mixing operations into scans. I'm wondering about how to handle
# up/down-scaling, and whether it's possible to convert both down and up scaled
# layers to a scan with the same sequence length as the data, without altering
# semantics.
#  - For down-scaled layers, change the input-output representation to a
#    data-scaled array, and make the scan only do computation conditional on
#    the iteration number. For example, if the inputs/outputs are down-scaled
#    by a factor of 2, and the data has a length 2L, the down-scaled layer in
#    user-code has length L (for both inputs and outputs). We convert that to a
#    scan over a sequence of length 2L, where each output (and input?) are
#    duplicated (or zero?), and there is a cond which decides whether
#    computation actually occurs, depending on whether the iteration number is
#    even or odd.
#  - For up-scaled layers, we reshape, introducing an extra dimension with size
#    equal to the upscaling factor.

# After the above, we make the assumptions that
#  - the only sequence-mixing operations in the function are scans, except for
#    a single causal pad, with padding of 1 at the start and -1 at the end of
#    the sequence axis.
#  - All intermediates have the same length along the sequence axis.

# I'm worried the above will be quite brittle, but it is reasonably simple and
# efficient...

# Maybe one should assume that the single element causal pad takes place
# outside of the function being transformed, and then our task is somewhat
# simplified --- take a function where every sequence mixing operation is
# expressed as a scan, all arrays have the same seq length, and convert it to a
# single scan update...

# Another elegant way to frame the scan thing. The data-flow/dependency in the
# functions we're working with is valid if and only if the function can be
# expressed as a scan.

# Notes 13/5/2025
# I realised that for the scan scanify rule, we need the initial carry. We can,
# I think, get it by doing some evaluation in make_init_carry. If we just setup
# an environment with the jaxpr's constvars in it, have a maybe_read function
# which returns a variables value if it is in env, otherwise returns its aval,
# and for equations with no scan invars actually evaluate the primitive, I
# think that should hopefully cover most reasonable use-cases.

# Notes 16/5/2025
# Just hit an issue because the rules are currently applied during init and
# during the body fun, with differently shaped scanned arguments (during body
# fun the scanned axis has been removed). I'm not wondering whether we should
# refactor the core to have one function doing what make_body and
# make_init_carry currently do. This could do an an initial pass through the
# jaxpr building up the init carry and a list of local body funs, then create a
# global body fun which iterates through the list. This would mean that each
# rule is only evaluated once, on fully-shaped arguments.
import jax.numpy as jnp
from jax import lax, make_jaxpr
from fastar.api import as_scan

import jax.numpy as jnp
from jax import lax

y = jnp.array(1.)
#                    o, i, w
conv_rhs = jnp.ones((3, 1, 2))

def f(xs):
    #ys = jnp.expand_dims(ys, 0)
    #zs = lax.conv_general_dilated(
    #    ys, conv_rhs, [1], [[1, 0]],
    #    dimension_numbers=("NLC", "OIL", "NLC"),
    #)

    xs = lax.broadcast_in_dim(
        xs, (3, 2), (0, 1)
    )
    bs = lax.scan(lambda c, x: (y + c + x, c * x), jnp.array([1., 2.]), xs)[1]
    cs = 2 + bs
    ds = lax.reduce_sum(cs, [1])
    return ds

xs = jnp.array([[1.],
                [3.],
                [5.]])

f_jaxpr = make_jaxpr(f)(xs)
body_fn, init_carry = as_scan(f, xs)

def f_scan(xs):
    out_carry, ans = lax.scan(body_fn, init_carry, xs)
    return ans

f_scan(xs)
