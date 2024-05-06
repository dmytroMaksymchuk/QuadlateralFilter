import math

import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel(sigma, x):
    return (1 / (2 * math.pi) * sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))


def gaussian_filter(img, sigma):
    # Convert to grayscale opencv

    kernel_size = math.ceil(3 * sigma)
    rows = img.shape[0]
    cols = img.shape[1]
    blured_img = np.zeros_like(img)

    # Generate grid of distances
    X, Y = np.meshgrid(np.arange(-kernel_size, kernel_size + 1), np.arange(-kernel_size, kernel_size + 1))
    distances = np.sqrt(X ** 2 + Y ** 2)

    # Generate gaussian kernel
    kernel = gaussian_kernel(sigma, distances)

    # Normalize kernel
    kernel /= np.sum(kernel)

    # Apply filter
    for i in range(rows):
        for j in range(cols):
            region = img[max(0, i - kernel_size):min(rows, i + kernel_size + 1),
                     max(0, j - kernel_size):min(cols, j + kernel_size + 1)]
            blured_img[i, j] = np.sum(region * kernel[:region.shape[0], :region.shape[1]])

    return blured_img
