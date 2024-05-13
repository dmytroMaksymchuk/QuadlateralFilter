import math

import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots

from QuadlateralFilter.bilateral.bilateralFilter import bilateral_filter
from QuadlateralFilter.helpers.addGaussianNoise import add_gauss_noise_2d
from QuadlateralFilter.trilateral.trilateralFilter2D import match_shape, distance, gaussian_kernel, trilateral_filter_2d


def gaussian_kernel_1d(sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))


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
            grad_s_kernel = gaussian_kernel_1d(sigma_intensity, np.abs(region_gradient_y))

            # Kernel
            grad_kernel = grad_s_kernel * spatial_kernel[
                                          max(kernel_size - i, 0):min(kernel_size + (img.shape[0] - i), (2 * kernel_size) + 1),
                                          max(kernel_size - j, 0):min(kernel_size + (img.shape[1] - j), (2 * kernel_size) + 1)]

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
            grad_s_kernel = gaussian_kernel(sigma_intensity, np.abs(region_gradient_x))

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
    beta = 5
    sigma_intensity = beta * distance(np.max(np.sqrt(average_gradients_y ** 2 + average_gradients_x ** 2)),
                                      np.min(np.sqrt(average_gradients_y ** 2 + average_gradients_x ** 2)))
    # R = sigma_intensity
    print("sigma_intensity", sigma_intensity)

    derivative_y, derivative_x = get_bilateral_derivative(img, kernel_size, spatial_kernel, sigma_intensity)
    derivative_xy = get_bilateral_derivative_y(derivative_x, kernel_size, spatial_kernel, sigma_intensity)
    derivative_xx = get_bilateral_derivative_x(derivative_x, kernel_size, spatial_kernel, sigma_intensity)
    derivative_yy = get_bilateral_derivative_y(derivative_y, kernel_size, spatial_kernel, sigma_intensity)

    quad_planes = np.empty((img.shape[1], img.shape[0]), dtype=object)

    # Apply filter
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Define region of interest
            regionLB = (max(0, i - kernel_size), max(0, j - kernel_size))
            regionUB = (min(img.shape[0], i + kernel_size + 1), min(img.shape[1], j + kernel_size + 1))
            region = img[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]].astype(np.float32)

            y_arranged = (np.arange(regionLB[0], regionUB[0]) - i)
            x_arranged = (np.arange(regionLB[1], regionUB[1]) - j)


            # Compute plane ax^2 + by^2 + cx + dy + e
            # Q_f(x, y) = f(x_0, y_0) + f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0) + ½ f_{xx}(x_0, y_0)(x-x_0)^2 + f_{xy}(x_0, y_0)(x-x_0)(y-y_0) + ½ f_{yy}(x_0, y_0)(y-y_0)^2

            change_X = np.repeat(x_arranged[np.newaxis, :], len(y_arranged), axis=0)
            change_Y = np.repeat(y_arranged[:, np.newaxis], len(x_arranged), axis=1)

            quad_plane = derivative_x[i][j] * change_X + derivative_y[i][j] * change_Y
            quad_plane += img[i][j]
            quad_plane += 0.5 * derivative_xx[i][j] * change_X ** 2 + 0.5 * derivative_yy[i][j] * change_Y ** 2 + derivative_xy[i][j] * change_X * change_Y
            quad_planes[i][j] = quad_plane

            # I{delta}(x, vector)
            diff_from_plane = region - quad_plane

            # S-kernel
            s_kernel = gaussian_kernel(sigma_intensity, np.abs(diff_from_plane))

            # Neighborhood inclusion kernel
            # inclusion = distance(
            #     tilting_vectors_y[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]] - tilting_vectors_y[i][j],
            #     tilting_vectors_x[regionLB[0]:regionUB[0], regionLB[1]:regionUB[1]] - tilting_vectors_x[i][j]) < R

            # Kernel
            kernel = s_kernel * spatial_kernel[
                                max(kernel_size - i, 0):min(kernel_size + (img.shape[0] - i), (2 * kernel_size) + 1),
                                max(kernel_size - j, 0):min(kernel_size + (img.shape[1] - j),
                                                            (2 * kernel_size) + 1)]
            kernel /= np.sum(kernel)

            # Compute final value
            filtered_img[i][j] = img[i][j] + np.sum(diff_from_plane * kernel)

    return filtered_img, quad_planes



def add_gauss_noise_shape(img, sigma=1):
    # Compute image dimensions
    rows, cols = img.shape
    img = img.astype(np.float32)
    # Add Gaussian noise
    noise = np.random.normal(0, sigma, (rows, cols))

    noisy_img = img + noise

    return noisy_img


import plotly.graph_objects as go
if __name__ == '__main__':
    # Define grid of x and y coordinates
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.sin(X) * np.cos(Y) * 10

    # noised_Z = add_gauss_noise_shape(Z, sigma=1)
    noised_Z = Z

    tri_filtered_Z = trilateral_filter_2d(noised_Z, 5)

    filtered_Z, quad = quadrateral_filter_2d(noised_Z, 5)

    # Visualize ------------------------

    fig = go.Figure(data=[go.Surface(z = noised_Z)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='Quadlateral filter', autosize=True,
                      width=1000, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

    fig2 = go.Figure(data=[go.Surface(z=filtered_Z)])
    fig2.update_traces(contours_z=dict(show=True, usecolormap=True,
                                      highlightcolor="limegreen", project_z=True))
    fig2.update_layout(title='Quadlateral filter', autosize=True,
                      width=1000, height=800,
                      margin=dict(l=65, r=50, b=65, t=90))
    fig2.show()

    fig3 = go.Figure(data=[go.Surface(z=tri_filtered_Z)])
    fig3.update_traces(contours_z=dict(show=True, usecolormap=True,
                                       highlightcolor="limegreen", project_z=True))
    fig3.update_layout(title='Quadlateral filter', autosize=True,
                       width=1000, height=800,
                       margin=dict(l=65, r=50, b=65, t=90))
    fig3.show()

    print("Mean diff quad", np.mean(np.abs(Z - filtered_Z)))
    print("Mean tri quad", np.mean(np.abs(Z - tri_filtered_Z)))
