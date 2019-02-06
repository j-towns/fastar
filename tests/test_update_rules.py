import jax.lax as lax
import numpy.random
from fastar.test_util import check_fun

rng = numpy.random.RandomState(0)

R = rng.randn


# Unops
def test_sin(): check_fun(rng, lax.sin, R(1, 2))

# Binops
def test_add(): check_fun(rng, lax.add, R(1, 2), R(3, 1))
def test_sub(): check_fun(rng, lax.sub, R(1, 2), R(3, 1))
def test_mul(): check_fun(rng, lax.mul, R(1, 2), R(3, 1))
