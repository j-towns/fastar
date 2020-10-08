from jax import config

config.enable_omnistaging()

from . import rules
from .jaxpr_util import tie_the_knot
from .api import lazy_eval, lazy_eval_fixed_point
