import math

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from helpers.addGaussianNoise import add_gauss_noise_1d, add_gauss_noise_2d
from bilateral.bilateralFilter1D import bilateral_filter_1D

def gaussian_kernel(sigma, x):
    return (1 / (2 * math.pi * sigma ** 2)) * np.exp(-(x ** 2) / (2 * sigma ** 2))

def distance(x, y):
    return np.sqrt(x ** 2 + y ** 2)
def min_tuple(t1, t2):
    return tuple(map(lambda x, y: min(x, y), t1, t2))
def match_shape(arr, shape, val):
    new_arr = np.full(shape, val)
    new_arr[:arr.shape[0], :arr.shape[1]] = arr
    return new_arr

def trilateral_filter_2d(img, sigma_spatial):
    # Initialize filtered image
    # Compute image dimensions
    rows, cols = img.shape

    # Initialize filtered image
    filtered_img = np.zeros_like(img)
    print(img.shape)

    # Generate spatial kernel
    kernel_size = math.ceil(3 * sigma_spatial)
    X, Y = np.meshgrid(np.arange(-kernel_size, kernel_size + 1), np.arange(-kernel_size, kernel_size + 1))
    spatial_kernel = gaussian_kernel(sigma_spatial, np.sqrt(X ** 2 + Y ** 2))

    # Compute average gradients in the neighborhood for each pixel
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



    # Compute Sigma for range kernel
    beta = 1
    sigma_intensity = beta * distance(np.max(np.sqrt(average_gradients_y ** 2 + average_gradients_x ** 2)),
                                      np.min(np.sqrt(average_gradients_y ** 2 + average_gradients_x ** 2)))
    R = sigma_intensity
    print("sigma_intensity", sigma_intensity)

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
            grad_kernel = grad_s_kernel * spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (img.shape[0] - i), (2 * kernel_size) + 1),
                          max(kernel_size - j, 0):min(kernel_size + (img.shape[1] - j), (2 * kernel_size) + 1)]

            if np.sum(grad_kernel) != 0:
                grad_kernel /= np.sum(grad_kernel)

            # Compute tilting vector G
            tilting_vectors_y[i][j] = np.sum(region_gradient_y * grad_kernel)
            tilting_vectors_x[i][j] = np.sum(region_gradient_x * grad_kernel)

    # Apply filter
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Define region of interest
            regionLB = (max(0, i - kernel_size), max(0, j - kernel_size))
            regionUB = (min(img.shape[0], i + kernel_size + 1), min(img.shape[1], j + kernel_size + 1))
            region = img[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]].astype(np.float32)

            y_arranged = (np.arange(regionLB[0], regionUB[0]) - i)
            x_arranged = (np.arange(regionLB[1], regionUB[1]) - j)

            # Compute plane
            plane = tilting_vectors_y[i][j] * np.repeat(y_arranged[:, np.newaxis], len(x_arranged), axis=1)
            plane += tilting_vectors_x[i][j] * np.repeat(x_arranged[np.newaxis, :], len(y_arranged), axis=0)
            plane += img[i][j]
            # I{delta}(x, vector)
            diff_from_plane = region - plane

            # S-kernel
            s_kernel = gaussian_kernel(sigma_intensity, np.abs(diff_from_plane))

            # Neighborhood inclusion kernel
            inclusion = distance(tilting_vectors_y[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]] - tilting_vectors_y[i][j],
                                 tilting_vectors_x[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]] - tilting_vectors_x[i][j]) < R

            # Kernel
            kernel = s_kernel * spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (img.shape[0] - i), (2 * kernel_size) + 1),
                          max(kernel_size - j, 0):min(kernel_size + (img.shape[1] - j), (2 * kernel_size) + 1)] * inclusion
            kernel /= np.sum(kernel)

            # Compute final value
            filtered_img[i][j] = img[i][j] + np.sum(diff_from_plane * kernel)

    return filtered_img


if __name__ == '__main__':
    img = cv.imread('../images/smaller_lamp.jpg', cv.IMREAD_GRAYSCALE)
    sigmaSpatial = 3
    noised_img = add_gauss_noise_2d(img, 6)
    filtered_image = trilateral_filter_2d(noised_img, sigmaSpatial)
    # filtered_image = cv.bilateralFilter(noised_img, 6, 20, 10)

    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.subplot(132)
    plt.imshow(noised_img, cmap='gray')
    plt.title('Noised Image')
    plt.subplot(133)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')

    plt.show()

    diff_noised = np.abs(img.astype(np.float32) - noised_img)
    diff_filtered = np.abs(img.astype(np.float32) - filtered_image)
    print('Noised Image')
    print('Max difference:', np.max(diff_noised))
    print('Min difference:', np.min(diff_noised))
    print('Mean difference:', np.mean(diff_noised))
    print('Filtered Image')
    print('Max difference:', np.max(diff_filtered))
    print('Min difference:', np.min(diff_filtered))
    print('Mean difference:', np.mean(diff_filtered))
