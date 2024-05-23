import numpy as np


def distance(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def min_tuple(t1, t2):
    return tuple(map(lambda x, y: min(x, y), t1, t2))


def match_shape(arr, shape, val):
    new_arr = np.full(shape, val)
    new_arr[:arr.shape[0], :arr.shape[1]] = arr
    return new_arr
