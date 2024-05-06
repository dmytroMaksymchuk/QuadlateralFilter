import math

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from helpers.addGaussianNoise import add_gauss_noise_1d


def gaussian_kernel_1d(sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))


def bilateral_filter_1D(inp, sigma_spatial, sigma_intensity):
    # Initialize filtered image
    filtered_inp = np.zeros_like(inp)

    # Generate spatial kernel
    kernel_size = math.ceil(3 * sigma_spatial)
    spatial_kernel = gaussian_kernel_1d(sigma_spatial, np.arange(-kernel_size, kernel_size + 1))

    # Apply filter
    for i in range(np.size(inp)):
        # Define region of interest
        region = inp[max(0, i - kernel_size):min(np.size(inp), i + kernel_size + 1)]
        region = region.astype(np.float32)

        # Compute intensity differences
        intensity_diff = region - inp[i]

        # Compute range kernel
        range_kernel = gaussian_kernel_1d(sigma_intensity, np.abs(intensity_diff))

        # Compute final kernel
        kernel = spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (inp.shape[0] - i), (2 * kernel_size) + 1)] * range_kernel

        # Normalize kernel
        kernel /= np.sum(kernel)

        # Apply weighted sum
        filtered_inp[i] = np.sum(region * kernel)

    return filtered_inp


if __name__ == '__main__':
    const1 = np.full(30, 20)
    const2 = np.full(30, 180)
    down = np.arange(200, 50, -10)

    # Create an array from 200 back to 10 (excluding 200)
    up = np.arange(50, 180, 50)

    # Concatenate the two arrays
    inp = np.concatenate((const1, down, up, const2))

    inp = add_gauss_noise_1d(inp, 5)

    out = bilateral_filter_1D(inp, 20, 20)

    out_cv = cv.bilateralFilter(inp, 6, 2, 2)

    plt.figure(figsize=(5, 5))
    plt.subplot(121)
    plt.plot(inp)
    plt.title('Original')

    plt.subplot(122)
    plt.plot(out)
    plt.title('Bilateral Filter')


    plt.show()

