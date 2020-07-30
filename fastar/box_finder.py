from warnings import warn
import numpy as np

import numba


def test_boxes(starts, sizes, dim):
    assert sizes[dim] == 1
    i = 1
    while True:
      yield tuple(start + i if d == dim else slice(start, start + size)
                  for d, (start, size) in enumerate(zip(starts, sizes)))
      i = i + 1

def box_finder(arr, look_for, switch_to):
  return box_finders.get(arr.ndim, box_finder_generic)(arr, look_for, switch_to)

def box_finder_generic(arr, look_for, switch_to):
  warn("Falling back on slow box finder for array with ndim > 6")
  it = np.nditer(arr, flags=['multi_index'])
  for k in it:
    if k == look_for:
      starts = it.multi_index
      arr[starts] = switch_to
      sizes = arr.ndim * [1]
      for d in range(arr.ndim):
        box_iter = test_boxes(starts, sizes, d)
        test_box = next(box_iter)
        while (starts[d] + sizes[d] < arr.shape[d]
               and np.all(arr[test_box] == look_for)):
          arr[test_box] = switch_to
          sizes[d] = sizes[d] + 1
          test_box = next(box_iter)
      yield starts, sizes

def box_finder0(arr, look_for, switch_to):
  if arr == look_for:
    arr[()] = switch_to
    yield (), ()

@numba.jit(nopython=True)
def box_finder1(arr, look_for, switch_to):
  I, = arr.shape
  for i in range(I):
    if arr[i] == look_for:
      arr[i] = switch_to
      i_test = i + 1
      while i_test < I and arr[i_test] == look_for:
        arr[i_test] = switch_to
        i_test += 1
      yield [i], [i_test - i]

@numba.jit(nopython=True)
def box_finder2(arr, look_for, switch_to):
  I, J = arr.shape
  for i in range(I):
    for j in range(J):
      if arr[i, j] == look_for:
        arr[i, j] = switch_to
        i_test = i + 1
        while i_test < I and arr[i_test, j] == look_for:
          arr[i_test, j] = switch_to
          i_test += 1
        j_test = j + 1
        while j_test < J and np.all(arr[i:i_test, j_test] == look_for):
          arr[i:i_test, j_test] = switch_to
          j_test += 1
        yield [i, j], [i_test - i, j_test - j]

@numba.jit(nopython=True)
def box_finder3(arr, look_for, switch_to):
  I, J, K = arr.shape
  for i in range(I):
    for j in range(J):
      for k in range(K):
        if arr[i, j, k] == look_for:
          arr[i, j, k] = switch_to
          i_test = i + 1
          while i_test < I and arr[i_test, j, k] == look_for:
            arr[i_test, j, k] = switch_to
            i_test += 1
          j_test = j + 1
          while j_test < J and np.all(arr[i:i_test, j_test, k] == look_for):
            arr[i:i_test, j_test, k] = switch_to
            j_test += 1
          k_test = k + 1
          while k_test < K and np.all(arr[i:i_test, j:j_test, k_test] == look_for):
            arr[i:i_test, j:j_test, k_test] = switch_to
            k_test += 1
          yield [i, j, k], [i_test - i, j_test - j, k_test - k]

@numba.jit(nopython=True)
def box_finder4(arr, look_for, switch_to):
  I, J, K, L = arr.shape
  for i in range(I):
    for j in range(J):
      for k in range(K):
        for l in range(L):
          if arr[i, j, k, l] == look_for:
            arr[i, j, k, l] = switch_to
            i_test = i + 1
            while i_test < I and arr[i_test, j, k, l] == look_for:
              arr[i_test, j, k, l] = switch_to
              i_test += 1
            j_test = j + 1
            while j_test < J and np.all(arr[i:i_test, j_test, k, l] == look_for):
              arr[i:i_test, j_test, k, l] = switch_to
              j_test += 1
            k_test = k + 1
            while k_test < K and np.all(arr[i:i_test, j:j_test, k_test, l] == look_for):
              arr[i:i_test, j:j_test, k_test, l] = switch_to
              k_test += 1
            l_test = l + 1
            while l_test < L and np.all(arr[i:i_test, j:j_test, k:k_test, l_test] == look_for):
              arr[i:i_test, j:j_test, k:k_test, l_test] = switch_to
              l_test += 1
            yield [i, j, k, l], [i_test - i, j_test - j, k_test - k, l_test - l]

@numba.jit(nopython=True)
def box_finder5(arr, look_for, switch_to):
  I, J, K, L, M = arr.shape
  for i in range(I):
    for j in range(J):
      for k in range(K):
        for l in range(L):
          for m in range(M):
            if arr[i, j, k, l, m] == look_for:
              arr[i, j, k, l, m] = switch_to
              i_test = i + 1
              while i_test < I and arr[i_test, j, k, l, m] == look_for:
                arr[i_test, j, k, l, m] = switch_to
                i_test += 1
              j_test = j + 1
              while j_test < J and np.all(arr[i:i_test, j_test, k, l, m] == look_for):
                arr[i:i_test, j_test, k, l, m] = switch_to
                j_test += 1
              k_test = k + 1
              while k_test < K and np.all(arr[i:i_test, j:j_test, k_test, l, m] == look_for):
                arr[i:i_test, j:j_test, k_test, l, m] = switch_to
                k_test += 1
              l_test = l + 1
              while l_test < L and np.all(arr[i:i_test, j:j_test, k:k_test, l_test, m] == look_for):
                arr[i:i_test, j:j_test, k:k_test, l_test, m] = switch_to
                l_test += 1
              m_test = m + 1
              while m_test < M and np.all(arr[i:i_test, j:j_test, k:k_test, l:l_test, m_test] == look_for):
                arr[i:i_test, j:j_test, k:k_test, l:l_test, m_test] = switch_to
                m_test += 1
              yield [i, j, k, l, m], [i_test - i, j_test - j, k_test - k, l_test - l, m_test - m]

@numba.jit(nopython=True)
def box_finder6(arr, look_for, switch_to):
  I, J, K, L, M, N = arr.shape
  for i in range(I):
    for j in range(J):
      for k in range(K):
        for l in range(L):
          for m in range(M):
            for n in range(N):
              if arr[i, j, k, l, m, n] == look_for:
                arr[i, j, k, l, m, n] = switch_to
                i_test = i + 1
                while i_test < I and arr[i_test, j, k, l, m, n] == look_for:
                  arr[i_test, j, k, l, m, n] = switch_to
                  i_test += 1
                j_test = j + 1
                while j_test < J and np.all(arr[i:i_test, j_test, k, l, m, n] == look_for):
                  arr[i:i_test, j_test, k, l, m, n] = switch_to
                  j_test += 1
                k_test = k + 1
                while k_test < K and np.all(arr[i:i_test, j:j_test, k_test, l, m, n] == look_for):
                  arr[i:i_test, j:j_test, k_test, l, m, n] = switch_to
                  k_test += 1
                l_test = l + 1
                while l_test < L and np.all(arr[i:i_test, j:j_test, k:k_test, l_test, m, n] == look_for):
                  arr[i:i_test, j:j_test, k:k_test, l_test, m, n] = switch_to
                  l_test += 1
                m_test = m + 1
                while m_test < M and np.all(arr[i:i_test, j:j_test, k:k_test, l:l_test, m_test, n] == look_for):
                  arr[i:i_test, j:j_test, k:k_test, l:l_test, m_test, n] = switch_to
                  m_test += 1
                n_test = n + 1
                while n_test < N and np.all(arr[i:i_test, j:j_test, k:k_test, l:l_test, m:m_test, n_test] == look_for):
                  arr[i:i_test, j:j_test, k:k_test, l:l_test, m:m_test, n_test] = switch_to
                  n_test += 1
                yield [i, j, k, l, m, n], [i_test - i, j_test - j, k_test - k, l_test - l, m_test - m, n_test - n]

box_finders = {}
box_finders[0] = box_finder0
box_finders[1] = box_finder1
box_finders[2] = box_finder2
box_finders[3] = box_finder3
box_finders[4] = box_finder4
box_finders[5] = box_finder5
box_finders[6] = box_finder6


def static_box_finder(arr, look_for=1):
  tmp = np.asarray(arr == look_for)
  return list(box_finder(tmp, 1, 0))
