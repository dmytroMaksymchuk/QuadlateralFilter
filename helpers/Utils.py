import math

import numpy as np


def distance(x, y):
    return np.sqrt(x ** 2 + y ** 2)


def min_tuple(t1, t2):
    return tuple(map(lambda x, y: min(x, y), t1, t2))


def match_shape(arr, shape, val):
    new_arr = np.full(shape, val)
    new_arr[:arr.shape[0], :arr.shape[1]] = arr
    return new_arr


def calc_psnr(original, restored):
    mse = np.mean((original.astype(np.float32) - restored.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calc_uncertainty(norm_factor, mean, std, alpha=2, beta=1):
    uncretainty = 1/(1 + math.e ** (alpha*((norm_factor - mean)/std) + beta))
    return uncretainty

