import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from QuadlateralFilter.helpers.gaussianHelper import gaussian_kernel_2d
from QuadlateralFilter.helpers.gradients import get_bilateral_derivative_y, get_bilateral_derivative_x


def quadrateral_filter_2d(img, sigma_spatial, sigma_intensity=10):
    # Initialize filtered image
    filtered_img = np.zeros_like(img).astype(np.float32)

    # Generate spatial kernel
    kernel_size = math.ceil(3 * sigma_spatial)
    X, Y = np.meshgrid(np.arange(-kernel_size, kernel_size + 1), np.arange(-kernel_size, kernel_size + 1))
    spatial_kernel = gaussian_kernel_2d(sigma_spatial, np.sqrt(X ** 2 + Y ** 2))

    # Compute derivatives used for reconstruction
    derivative_y = get_bilateral_derivative_y(img, kernel_size, spatial_kernel, sigma_intensity)
    derivative_x = get_bilateral_derivative_x(img, kernel_size, spatial_kernel, sigma_intensity)
    derivative_xy = get_bilateral_derivative_y(derivative_x, kernel_size, spatial_kernel, sigma_intensity)
    derivative_xx = get_bilateral_derivative_x(derivative_x, kernel_size, spatial_kernel, sigma_intensity)
    derivative_yy = get_bilateral_derivative_y(derivative_y, kernel_size, spatial_kernel, sigma_intensity)

    # Compare this plane to neighbouring pixels for range kernel
    quad_planes = np.empty((img.shape[0], img.shape[1]), dtype=object)

    # Apply filter
    k_vals = np.zeros_like(img).astype(np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Define region of interest
            regionLB = (max(0, i - kernel_size), max(0, j - kernel_size))
            regionUB = (min(img.shape[0], i + kernel_size + 1), min(img.shape[1], j + kernel_size + 1))
            region = img[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]].astype(np.float32)

            y_arranged = (np.arange(regionLB[0], regionUB[0]) - i)
            x_arranged = (np.arange(regionLB[1], regionUB[1]) - j)

            # Compute plane ax^2 + by^2 + cx + dy + e -> (Tyler's 2nd order approximation)

            change_X = np.repeat(x_arranged[np.newaxis, :], len(y_arranged), axis=0)
            change_Y = np.repeat(y_arranged[:, np.newaxis], len(x_arranged), axis=1)

            quad_plane = derivative_x[i][j] * change_X + derivative_y[i][j] * change_Y
            if i % 10 and j == 0 == 0:
                print(i)
            quad_plane += img[i][j]
            quad_plane += 0.5 * derivative_xx[i][j] * change_X ** 2 + 0.5 * derivative_yy[i][j] * change_Y ** 2 + \
                          derivative_xy[i][j] * change_X * change_Y
            quad_planes[i][j] = quad_plane

            # I{delta}(x, vector)
            diff_from_plane = region - quad_plane

            # S-kernel
            s_kernel = gaussian_kernel_2d(sigma_intensity, np.abs(diff_from_plane))

            # Kernel
            kernel = s_kernel * spatial_kernel[
                                max(kernel_size - i, 0):min(kernel_size + (img.shape[0] - i), (2 * kernel_size) + 1),
                                max(kernel_size - j, 0):min(kernel_size + (img.shape[1] - j),
                                                            (2 * kernel_size) + 1)]
            k_vals[i][j] = np.sum(kernel)
            kernel /= np.sum(kernel)

            # Compute final value
            filtered_img[i][j] = img[i][j] + np.sum(diff_from_plane * kernel)

    return filtered_img, quad_planes

if __name__ == '__main__':
    x = np.linspace(-4.5, 4.5, 100)
    y = np.linspace(-4.5, 4.5, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.sin(X) * np.cos(Y) * 200
    Z = Z.clip(0, 255).astype(np.uint8)

    sigmaSpatial = 3
    k_vals, trash = quadrateral_filter_2d(Z, sigmaSpatial)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.log10(k_vals), cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.show()
