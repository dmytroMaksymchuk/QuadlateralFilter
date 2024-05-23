import numpy as np

from QuadlateralFilter.helpers.Utils import match_shape
from QuadlateralFilter.helpers.gaussianHelper import gaussian_kernel_2d


def get_bilateral_derivative_y(img, kernel_size, spatial_kernel, sigma_intensity):
    tilting_vectors_y = np.zeros_like(img).astype(np.float32)
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

            region_gradient_y = match_shape(region_gradient_y, region.shape, gradient_point[0])

            # S-kernel
            grad_s_kernel = gaussian_kernel_2d(sigma_intensity, np.abs(region_gradient_y))

            # Kernel
            grad_kernel = grad_s_kernel * spatial_kernel[
                                          max(kernel_size - i, 0):min(kernel_size + (img.shape[0] - i),
                                                                      (2 * kernel_size) + 1),
                                          max(kernel_size - j, 0):min(kernel_size + (img.shape[1] - j),
                                                                      (2 * kernel_size) + 1)]

            if np.sum(grad_kernel) != 0:
                grad_kernel /= np.sum(grad_kernel)

            # Compute tilting vector G
            tilting_vectors_y[i][j] = np.sum(region_gradient_y * grad_kernel)

    return tilting_vectors_y


def get_bilateral_derivative_x(img, kernel_size, spatial_kernel, sigma_intensity):
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

            # x gradients, (shift right)
            shifted_region_x = img[regionLB[0]:regionUB[0],
                               regionLB[1] + 1:min(img.shape[1], regionUB[1] + 1)]

            if len(shifted_region_x[0]) == 0:
                region_gradient_x = 0
            else:
                region_gradient_x = shifted_region_x - region[:, :len(shifted_region_x[0])]

            # Match shape (in case of boundaries)
            region_gradient_x = match_shape(region_gradient_x, region.shape, gradient_point[1])

            # S-kernel
            grad_s_kernel = gaussian_kernel_2d(sigma_intensity, np.abs(region_gradient_x))

            # Kernel
            grad_kernel = grad_s_kernel * spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (img.shape[0] - i),
                                                                                     (2 * kernel_size) + 1),
                                          max(kernel_size - j, 0):min(kernel_size + (img.shape[1] - j),
                                                                      (2 * kernel_size) + 1)]

            if np.sum(grad_kernel) != 0:
                grad_kernel /= np.sum(grad_kernel)

            # Compute tilting vector G
            tilting_vectors_x[i][j] = np.sum(region_gradient_x * grad_kernel)

    return tilting_vectors_x


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
