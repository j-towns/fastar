import functools

import jax.lax as lax
import numpy as np
from fastar.test_util import check_fun

rng = np.random.RandomState(0)
R = rng.randn
check = functools.partial(check_fun, rng)

# Unops
def test_sin(): check(lax.sin, R(1, 2))

# Binops
def test_add_scalar(): check(lax.add, R(), R())
def test_add_scalar_as_numpy(): check(lax.add, np.float64(R()), np.float64(R()))
def test_add_scalar_as_array(): check(lax.add, np.array(R()), np.array(R()))
def test_add_scalar_int(): check(lax.add, 4, 7)
def test_add_vector(): check(lax.add, R(1), R(2))
def test_add_matrix(): check(lax.add, R(1, 2), R(3, 1))

def test_sub(): check(lax.sub, R(1, 2), R(3, 1))

def test_mul(): check(lax.mul, R(1, 2), R(3, 1))

def test_dot_vector_vector(): check(lax.dot, R(2), R(2))
def test_dot_matrix_vector(): check(lax.dot, R(3, 2), R(2))
def test_dot_matrix_matrix(): check(lax.dot, R(3, 2), R(2, 4))