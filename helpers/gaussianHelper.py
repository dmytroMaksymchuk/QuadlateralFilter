import math

import numpy as np


def gaussian_kernel_1d(sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))


def gaussian_kernel_2d(sigma, x):
    return (1 / (2 * math.pi * sigma ** 2)) * np.exp(-(x ** 2) / (2 * sigma ** 2))


def add_gauss_noise_1d_signal(inp, sigma=1):
    # Compute image dimensions
    inp = inp.astype(np.float32)
    # Add Gaussian noise
    noise = np.random.normal(0, sigma, np.size(inp))

    noisy_out = inp + noise

    return noisy_out


def add_gauss_noise_2d_signal(img, sigma=1):
    # Compute image dimensions
    rows, cols = img.shape
    img = img.astype(np.float32)
    # Add Gaussian noise
    noise = np.random.normal(0, sigma, (rows, cols))

    return img + noise


def add_gauss_noise_2d_image(img, sigma=1):
    # Compute image dimensions
    noised = add_gauss_noise_2d_signal(img, sigma)
    ret = np.clip(noised, 0, 255).astype(np.uint8)
    return ret
