import numpy as np


def test_boxes(starts, sizes, dim):
    assert sizes[dim] == 1
    i = 1
    while True:
      yield tuple(start + i if d == dim else slice(start, start + size)
                  for d, (start, size) in enumerate(zip(starts, sizes)))
      i = i + 1

def box_finder(arr, look_for, switch_to):
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

def static_box_finder(arr, look_for=1):
  tmp = np.asarray(arr == look_for)
  return list(box_finder(tmp, 1, 0))
