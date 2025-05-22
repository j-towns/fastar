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
from typing import Iterable, TypeVar

T1 = TypeVar("T1")
T2 = TypeVar("T2")


# Copied from jax/_src/util.py because jax.util.safe_map is deprecated
def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f"length mismatch: {list(map(len, args))}"
    return list(map(f, *args))

def unzip2(xys: Iterable[tuple[T1, T2]]
    ) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
    """Unzip sequence of length-2 tuples into two tuples."""
    # Note: we deliberately don't use zip(*xys) because it is lazily evaluated,
    # is too permissive about inputs, and does not guarantee a length-2 output.
    xs: list[T1] = []
    ys: list[T2] = []
    for x, y in xys:
        xs.append(x)
        ys.append(y)
    return tuple(xs), tuple(ys)

def safe_zip(*args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(zip(*args))
