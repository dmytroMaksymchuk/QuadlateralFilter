import math

import matplotlib.pyplot as plt
import numpy as np

from helpers.addGaussianNoise import add_gauss_noise_1d
from bilateral.bilateralFilter1D import bilateral_filter_1D
from trilateral.trilateralFilter1D import trilateral_filter
from trilateral.trilateralFilter2D import match_shape, distance, gaussian_kernel


def gaussian_kernel_1d(sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))


def get_bilateral_derivative(img, kernel_size, spatial_kernel, sigma_intensity):
    tilting_vectors_y = np.zeros_like(img).astype(np.float32)
    tilting_vectors_x = np.zeros_like(img).astype(np.float32)
    # Precompute gradient vectors
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Define region of interest
            regionLB = (max(0, i - kernel_size), max(0, j - kernel_size))
            regionUB = (min(img.shape[0], i + kernel_size + 1), min(img.shape[1], j + kernel_size + 1))
            region = img[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]].astype(np.float32)

            # delta Iin (y, x)
            gradient_point = (int(img[min(img.shape[0] - 1, i + 1), j]) - img[i, j],
                              int(img[i, min(img.shape[1] - 1, j + 1)]) - img[i, j])

            # delta Iin for region(X, Y)
            # y gradients, (shift down)
            shifted_region_y = img[regionLB[0] + 1:min(img.shape[0], regionUB[0] + 1),
                               regionLB[1]:regionUB[1]]

            if len(shifted_region_y) == 0:
                region_gradient_y = 0
            else:
                region_gradient_y = shifted_region_y - region[:len(shifted_region_y), :]

            # x gradients, (shift right)
            shifted_region_x = img[regionLB[0]:regionUB[0],
                               regionLB[1] + 1:min(img.shape[1], regionUB[1] + 1)]

            if len(shifted_region_x[0]) == 0:
                region_gradient_x = 0
            else:
                region_gradient_x = shifted_region_x - region[:, :len(shifted_region_x[0])]

            # Match shape (in case of boundaries)
            region_gradient_y = match_shape(region_gradient_y, region.shape, gradient_point[0])
            region_gradient_x = match_shape(region_gradient_x, region.shape, gradient_point[1])

            # S-kernel
            vector_distances = distance(region_gradient_y, region_gradient_x)
            grad_s_kernel = gaussian_kernel(sigma_intensity, vector_distances)

            # Kernel
            grad_kernel = grad_s_kernel * spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (img.shape[0] - i),
                                                                                     (2 * kernel_size) + 1),
                                          max(kernel_size - j, 0):min(kernel_size + (img.shape[1] - j),
                                                                      (2 * kernel_size) + 1)]

            if np.sum(grad_kernel) != 0:
                grad_kernel /= np.sum(grad_kernel)

            # Compute tilting vector G
            tilting_vectors_y[i][j] = np.sum(region_gradient_y * grad_kernel)
            tilting_vectors_x[i][j] = np.sum(region_gradient_x * grad_kernel)

    return tilting_vectors_x, tilting_vectors_y


def get_average_gradients(img, kernel_size):
    average_gradients_x = np.zeros_like(img).astype(np.float32)
    average_gradients_y = np.zeros_like(img).astype(np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Define region of interest
            regionLB = (max(0, i - kernel_size), max(0, j - kernel_size))
            regionUB = (min(img.shape[0], i + kernel_size + 1), min(img.shape[1], j + kernel_size + 1))
            region = img[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]].astype(np.float32)

            # y gradients, (shift down)
            shifted_region_y = img[regionLB[0] + 1:min(img.shape[0], regionUB[0] + 1),
                               regionLB[1]:regionUB[1]]

            if len(shifted_region_y) == 0:
                average_gradients_y[i][j] = 0
            else:
                region_gradient = shifted_region_y - region[:len(shifted_region_y), :]
                average_gradients_y[i][j] = np.mean(region_gradient)

            # x gradients, (shift right)
            shifted_region_x = img[regionLB[0]:regionUB[0],
                               regionLB[1] + 1:min(img.shape[1], regionUB[1] + 1)]

            if len(shifted_region_x) == 0:
                average_gradients_x[i][j] = 0
            else:
                region_gradient = shifted_region_x - region[:, :len(shifted_region_x[0])]
                average_gradients_x[i][j] = np.mean(region_gradient)

    return average_gradients_x, average_gradients_y


def quadrateral_filter_2d(img, sigma_spatial):
    # Initialize filtered image
    filtered_img = np.zeros_like(img).astype(np.float32)

    # Generate spatial kernel
    kernel_size = math.ceil(3 * sigma_spatial)
    X, Y = np.meshgrid(np.arange(-kernel_size, kernel_size + 1), np.arange(-kernel_size, kernel_size + 1))
    spatial_kernel = gaussian_kernel(sigma_spatial, np.sqrt(X ** 2 + Y ** 2))

    # Compute average gradients in the neighborhood for each pixel
    average_gradients_y, average_gradients_x = get_average_gradients(img, sigma_spatial)

    # Compute Sigma for range kernel
    beta = 1
    sigma_intensity = beta * distance(np.max(np.sqrt(average_gradients_y ** 2 + average_gradients_x ** 2)),
                                      np.min(np.sqrt(average_gradients_y ** 2 + average_gradients_x ** 2)))
    # R = sigma_intensity
    print("sigma_intensity", sigma_intensity)

    first_bilateral_derivative = get_bilateral_derivative(img, kernel_size, spatial_kernel, sigma_intensity)
    second_bilateral_derivative = get_bilateral_derivative(first_bilateral_derivative, kernel_size, spatial_kernel,
                                                           sigma_intensity)

    # Apply filter
    for i in range(np.size(inp)):
        # Define region of interest
        regionLB = max(0, i - kernel_size)
        regionUB = min(np.size(inp), i + kernel_size + 1)
        region = inp[regionLB:regionUB]
        region = region.astype(np.float32)

        # Compute plane
        quad_plane = (inp[i] + first_bilateral_derivative[i] * (np.arange(regionLB, regionUB) - i) +
                      second_bilateral_derivative[i] * (np.arange(regionLB, regionUB) - i) ** 2)

        # I{delta}(x, vector)
        diff_from_plane = region - quad_plane

        # S-kernel
        s_kernel = gaussian_kernel_1d(sigma_intensity, np.abs(diff_from_plane))

        # TODO: Neighborhood inclusion kernel
        # inclusion = np.abs(gradient_vectors[regionLB:regionUB] - gradient_vectors[i]) <= R

        # Kernel
        kernel = spatial_kernel[
                 max(kernel_size - i, 0):min(kernel_size + (inp.shape[0] - i), (2 * kernel_size) + 1)] * s_kernel
        kernel /= np.sum(kernel)

        # Compute final value
        filtered_inp[i] = inp[i] + np.sum(diff_from_plane * kernel)

    return filtered_inp


if __name__ == '__main__':
    print("test")