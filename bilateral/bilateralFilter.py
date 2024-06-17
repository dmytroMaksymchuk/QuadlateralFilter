import math

import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel(sigma, x):
    return (1 / (2 * math.pi * sigma ** 2)) * np.exp(-x ** 2 / (2 * sigma ** 2))

def bilateral_filter(img, sigma_spatial, sigma_intensity):
    # Compute image dimensions
    rows, cols = img.shape

    # Initialize filtered image
    filtered_img = np.zeros_like(img)

    # Generate spatial kernel
    kernel_size = math.ceil(1.5 * sigma_spatial)
    X, Y = np.meshgrid(np.arange(-kernel_size, kernel_size + 1), np.arange(-kernel_size, kernel_size + 1))
    spatial_kernel = gaussian_kernel(sigma_spatial, np.sqrt(X ** 2 + Y ** 2))

    # Apply filter
    for i in range(rows):
        for j in range(cols):
            # Define region of interest
            region = img[max(0, i - kernel_size):min(rows, i + kernel_size + 1),
                         max(0, j - kernel_size):min(cols, j + kernel_size + 1)]
            region = region.astype(np.float32)  # Convert to float for precision

            # Compute intensity differences
            intensity_diff = region - img[i, j]

            # Compute range kernel
            range_kernel = gaussian_kernel(sigma_intensity, np.abs(intensity_diff))

            # Compute final kernel
            kernel = spatial_kernel[:region.shape[0], :region.shape[1]] * range_kernel

            # Normalize kernel
            kernel /= np.sum(kernel)

            # Apply weighted sum
            filtered_img[i, j] = np.sum(region * kernel)

    return filtered_img