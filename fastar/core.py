from typing import Any

from jax.extend.core import (
    ClosedJaxpr, Jaxpr, Primitive, Var, Literal, JaxprEqn
)
from jax.core import Atom
from jax import make_jaxpr, ShapeDtypeStruct
from jax.lax import scan_p

from fastar.util import safe_map


map = safe_map

scanify_rules = {}

def register_scanify_rule(p: Primitive, rule):
    scanify_rules[p] = rule

###############################################################################
# This section is copied from jax._src.core

# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def last_used(jaxpr: Jaxpr) -> dict[Var, JaxprEqn | None]:
    """Returns a mapping from every var in jaxpr to what equation uses it
    last."""
    last_used: dict[Var, JaxprEqn | None] = {
        v: None for v in jaxpr.outvars if not isinstance(v, Literal)}
    for eqn in reversed(jaxpr.eqns):
        for v in eqn.invars:
            if not isinstance(v, Literal) and v not in last_used:
                last_used[v] = eqn
    return last_used

def clean_up_dead_vars(eqn: JaxprEqn, env: dict[Var, Any],
                       last_used: dict[Var, JaxprEqn | None]):
    """Remove all eqn.invars from env if eqn is the last time they were
    used."""
    for v in {v for v in eqn.invars if not isinstance(v, Literal)}:
        if last_used[v] is eqn:
            # Delete ref to variable when it is no longer needed by next
            # equations.
            del env[v]
###############################################################################

# Need this wrapper type because the JAX AbstractValue type is not publicly
# exported
class Abstract:
    def __init__(self, aval):
        self.aval = aval

    @property
    def shape(self):
        return self.aval.shape

    @property
    def dtype(self):
        return self.aval.dtype

    @property
    def ndim(self):
        return self.aval.ndim

class Deleted:
    pass
deleted = Deleted()

def make_scan(j: ClosedJaxpr):
    jaxpr = j.jaxpr
    carry_init = []
    body_fns_ = []
    outscanvarss_ = []

    env = {}

    def write(v: Var, val: Any) -> None:
        env[v] = val

    def maybe_read(v: Atom) -> Any:
        if isinstance(v, Literal):
            return v.val
        elif v in env:
            if isinstance(env[v], Deleted):
                raise ScanConversionError(
                    "Using scan carry output is not supported"
                )
            else:
                return env[v]
        else:
            return Abstract(v.aval)
    map(write, jaxpr.constvars, j.consts)

    # Map from Var to scan axis
    scanvars = dict(zip(jaxpr.invars, len(jaxpr.invars) * [0]))
    for e in jaxpr.eqns:
        subfuns, bind_params = e.primitive.get_bind_params(e.params)

        inscanvars = [
            (i, scanvars[v]) for (i, v) in enumerate(e.invars) if type(v)
            is Var and v in scanvars
        ]
        if inscanvars:
            # TODO: Raise NotImplementedError if rule isn't defined
            in_avals = map(maybe_read, e.invars)
            init, body_fn, outscanvars, to_delete = (
                scanify_rules[e.primitive](
                    inscanvars, *subfuns, *in_avals, **bind_params
                )
            )
            outscanvars = [(e.outvars[i], l) for i, l in outscanvars]
            to_delete = [e.outvars[i] for i in to_delete]
            map(write, to_delete, len(to_delete) * [deleted])
            scanvars.update(outscanvars)
            carry_init.append(init)
            body_fns_.append(body_fn)
            outscanvarss_.append(outscanvars)
        else:
            in_vals = map(maybe_read, e.invars)
            if not any(isinstance(v, Abstract) for v in in_vals):
                subfuns, bind_params = e.primitive.get_bind_params(e.params)
                ans = e.primitive.bind(
                    *subfuns, *in_vals, **bind_params
                )
                if e.primitive.multiple_results:
                    map(write, e.outvars, ans)
                else:
                    write(e.outvars[0], ans)

    if any(o not in scanvars for o in jaxpr.outvars):
        # TODO: More detail here...
        raise ScanConversionError(
            "All of the outputs of the transformed function must be "
            "scanned over."
        )
    if any(scanvars[o] != 0 for o in jaxpr.outvars):
        # TODO: ...and here.
        raise ScanConversionError(
            "All outputs of the transformed function must be scanned over "
            "axis 0."
        )

    def body_fn(carry, xs):
        body_fns = list(reversed(body_fns_))
        carry_old = list(reversed(carry))
        outscanvarss = list(reversed(outscanvarss_))
        carry_new = []

        env: dict[Var, Any] = {}
        def read(v: Atom) -> Any:
            return v.val if isinstance(v, Literal) else env[v]

        def write(v: Var, val: Any) -> None:
            env[v] = val

        map(write, jaxpr.constvars, j.consts)
        map(write, jaxpr.invars, xs)
        lu = last_used(jaxpr)

        for e in jaxpr.eqns:
            subfuns, bind_params = e.primitive.get_bind_params(e.params)

            inscanvars = [
                (i, scanvars[v]) for (i, v) in enumerate(e.invars) if type(v)
                is Var and v in scanvars
            ]
            if inscanvars:
                carry_in = carry_old.pop()
                body_fn = body_fns.pop()
                outscanvars = outscanvarss.pop()
                in_vals = map(read, e.invars)
                # TODO: Raise NotImplementedError if rule isn't defined
                carry_out, ans = body_fn(carry_in, *in_vals)
                carry_new.append(carry_out)
                scanvars.update(outscanvars)
            else:
                ans = e.primitive.bind(
                    *subfuns, *map(read, e.invars), **bind_params
                )

            if e.primitive.multiple_results:
                map(write, e.outvars, ans)
            else:
                write(e.outvars[0], ans)
            clean_up_dead_vars(e, env, lu)
        if any(o not in scanvars for o in jaxpr.outvars):
            # TODO: More detail here...
            raise ScanConversionError(
                "All of the outputs of the transformed function must be "
                "scanned over."
            )
        if any(scanvars[o] != 0 for o in jaxpr.outvars):
            # TODO: ...and here.
            raise ScanConversionError(
                "All outputs of the transformed function must be scanned over "
                "axis 0."
            )
        return carry_new, map(read, jaxpr.outvars)
    return body_fn, carry_init

class ScanConversionError(Exception):
    pass
