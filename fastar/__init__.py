from jax import config

config.enable_omnistaging()

from . import rules
from .jaxpr_util import tie_the_knot, submerge_consts, inf
from .api import lazy_eval, lazy_eval_fixed_point
from .core import LazyArray
