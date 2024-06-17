import math

import matplotlib.pyplot as plt
import numpy as np

from bilateral.bilateralFilter1D import bilateral_filter_1D
from helpers.gaussianHelper import add_gauss_noise_1d_signal
from trilateral.trilateralFilter1D import trilateral_filter


def gaussian_kernel_1d(sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(-x ** 2 / (2 * sigma ** 2))

def get_bilateral_derivative(inp, kernel_size, spatial_kernel, sigma_intensity):

    gradient_vectors = np.zeros_like(inp).astype(np.float32)
    # Precompute gradient vectors
    for i in range(np.size(inp)):
        # Define region of interest
        regionLB = max(0, i - kernel_size)
        regionUB = min(np.size(inp), i + kernel_size + 1)
        region = inp[regionLB:regionUB]
        region = region.astype(np.float32)

        # delta Iin (x)
        gradient_x = inp[min(np.size(inp) - 1, i + 1)] - inp[i]
        # delta Iin for region(X)
        shifted_region = inp[regionLB + 1:min(np.size(inp), regionUB + 1)]

        gradient_region = shifted_region - region[:len(shifted_region)]

        # S-kernel
        grad_s_kernel = gaussian_kernel_1d(sigma_intensity, np.abs(gradient_region - gradient_x))

        # Kernel
        grad_kernel = spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (inp.shape[0] - i) - 1, (2 * kernel_size) + 1)] * grad_s_kernel

        if np.sum(grad_kernel) != 0:
            grad_kernel /= np.sum(grad_kernel)

        # Compute tilting vector G
        gradient_vectors[i] = np.sum(gradient_region * grad_kernel)

    return gradient_vectors

def get_average_gradients(inp, kernel_size):
    average_gradients = np.zeros_like(inp).astype(np.float32)
    for i in range(np.size(inp)):
        # Define region of interest
        regionLB = max(0, i - kernel_size)
        regionUB = min(np.size(inp), i + kernel_size + 1)
        region = inp[regionLB:regionUB]
        region = region.astype(np.float32)

        # delta Iin for region(X)
        shifted_region = inp[regionLB + 1:min(np.size(inp), regionUB + 1)]

        if len(shifted_region) < len(region):
            region_gradient = shifted_region - region[:len(shifted_region)]
        else:
            region_gradient = shifted_region - region

        # Compute average gradient
        average_gradients[i] = np.mean(region_gradient)
    return average_gradients

def quadrateral_filter(inp, sigma_spatial, sigma_intensity):
    # Initialize filtered image
    filtered_inp = np.zeros_like(inp).astype(np.float32)
    inp = inp.astype(np.float32)

    # Generate spatial kernel
    kernel_size = math.ceil(sigma_spatial * 1.5)
    spatial_kernel = gaussian_kernel_1d(sigma_spatial, np.arange(-kernel_size, kernel_size + 1))

    # Compute average gradients in the neighborhood for each pixel
    average_gradients = get_average_gradients(inp, sigma_spatial)

    # Compute Sigma for range kernel
    # beta = 7
    # sigma_intensity = beta * np.abs(np.max(average_gradients) - np.min(average_gradients))
    R = sigma_intensity

    first_bilateral_derivative = get_bilateral_derivative(inp, kernel_size, spatial_kernel, sigma_intensity)
    second_bilateral_derivative = get_bilateral_derivative(first_bilateral_derivative, kernel_size, spatial_kernel, sigma_intensity)

    # Apply filter
    line = np.zeros_like(16)
    for i in range(np.size(inp)):
        # Define region of interest
        regionLB = max(0, i - kernel_size)
        regionUB = min(np.size(inp), i + kernel_size + 1)
        region = inp[regionLB:regionUB]
        region = region.astype(np.float32)

        changeX = (np.arange(regionLB, regionUB) - i)

        quad_plane = inp[i] + first_bilateral_derivative[i] * changeX + 0.5 * second_bilateral_derivative[i] * (changeX ** 2)

        diff_from_plane = region - quad_plane

        # S-kernel
        s_kernel = gaussian_kernel_1d(sigma_intensity, np.abs(diff_from_plane))

        inclusion = np.abs(first_bilateral_derivative[regionLB:regionUB] - first_bilateral_derivative[i]) <= R

        # Kernel
        kernel = spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (inp.shape[0] - i), (2 * kernel_size) + 1)] * s_kernel * inclusion
        kernel /= np.sum(kernel)

        # Compute final value
        filtered_inp[i] = inp[i] + np.sum(diff_from_plane * kernel)

        if(i == 100):
            line = quad_plane


    return filtered_inp, line

if __name__ == '__main__':

    # Define the range of x values
    num_points = 120
    padding = 5
    x_values = np.linspace(-1, 1, num_points).astype(np.float32)  # Adjust the range and number of points as needed

    # Calculate the y values using the parabolic equation
    padd = np.zeros(padding)
    # y_p1 = x_values[:num_points//3] + 2
    # y_p2 = x_values[num_points//3:num_points//3*2] * 0 + 1
    # y_p3 = (x_values[num_points//3*2:] - 1) ** 4

    y_p1 = x_values[:num_points//4] * 0
    y_p2 = x_values[num_points//4:num_points//2] * 2 + 1
    y_p3 = (x_values[num_points//2:]-1) ** 6

    y_values = np.concatenate((y_p1, y_p2, y_p3))

    # Concatenate the two arrays
    inp_original = y_values
    inp = add_gauss_noise_1d_signal(inp_original, 0.01)

    sigma_spatial = 8
    sigma_intensity = 0.06


    out, line = quadrateral_filter(inp, sigma_spatial, sigma_intensity)
    out_bilat = bilateral_filter_1D(inp, sigma_spatial, sigma_intensity)
    out_trilat = trilateral_filter(inp, sigma_spatial, sigma_intensity)

    diff = np.abs(out - out_trilat)

    out = out[5:-5]
    out_bilat = out_bilat[5:-5]
    out_trilat = out_trilat[5:-5]
    inp = inp[5:-5]
    inp_original = inp_original[5:-5]


    plt.figure(figsize=(50, 5))
    plt.subplot(161)
    plt.plot(inp_original)
    plt.title('Original Input')

    # padd_size = 85
    # padd = np.zeros(padd_size)
    plt.subplot(162)
    plt.plot(inp)
    plt.title('Noised Input')

    plt.subplot(163)
    plt.plot(out)
    plt.legend(['quad', 'trilat'])
    plt.title('Quad Filter')

    plt.subplot(164)
    plt.plot(out_trilat)
    plt.title('Trilateral Filter')

    plt.subplot(165)
    plt.plot(out_bilat)
    plt.title('bilat')

    plt.subplot(166)
    plt.plot(out)
    plt.plot(out_trilat)
    plt.legend(['quad', 'trilat'])
    plt.title('Quad vs Trilat')




    plt.show()

    # Calculate average per pixel difference between the two outputs and original input
    error_quad = np.mean(np.abs(inp_original - out))
    error_bilateral = np.mean(np.abs(inp_original - out_bilat))
    print(f'Average per pixel difference between original and quad output: {error_quad}')
    print(f'Average per pixel difference between original and bilateral output: {error_bilateral}')


