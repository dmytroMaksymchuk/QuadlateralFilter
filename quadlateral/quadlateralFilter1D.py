import math

import matplotlib.pyplot as plt
import numpy as np

from helpers.addGaussianNoise import add_gauss_noise_1d
from bilateral.bilateralFilter1D import bilateral_filter_1D
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

        if len(shifted_region) < len(region):
            gradient_region = shifted_region - region[:len(shifted_region)]
        else:
            gradient_region = shifted_region - region

        # S-kernel
        grad_s_kernel = gaussian_kernel_1d(sigma_intensity, np.abs(gradient_region - gradient_x))

        # Kernel
        grad_kernel = spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (inp.shape[0] - i) - 1, (2 * kernel_size) + 1)] * grad_s_kernel

        if np.sum(grad_kernel) != 0:
            grad_kernel /= np.sum(grad_kernel)

        # Compute tilting vector G
        gradient_vectors[i] = round(np.sum(gradient_region * grad_kernel), 2)

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

def quadrateral_filter(inp, sigma_spatial):
    # Initialize filtered image
    filtered_inp = np.zeros_like(inp).astype(np.float32)
    inp = inp.astype(np.float32)

    # Generate spatial kernel
    kernel_size = math.ceil(sigma_spatial * 3)
    spatial_kernel = gaussian_kernel_1d(sigma_spatial, np.arange(-kernel_size, kernel_size + 1))

    # Compute average gradients in the neighborhood for each pixel
    average_gradients = get_average_gradients(inp, sigma_spatial)

    # Compute Sigma for range kernel
    beta = 7
    sigma_intensity = beta * np.abs(np.max(average_gradients) - np.min(average_gradients))

    first_bilateral_derivative = get_bilateral_derivative(inp, kernel_size, spatial_kernel, sigma_intensity)
    second_bilateral_derivative = get_bilateral_derivative(first_bilateral_derivative, kernel_size, spatial_kernel, sigma_intensity)

    # Apply filter
    for i in range(np.size(inp)):
        # Define region of interest
        regionLB = max(0, i - kernel_size)
        regionUB = min(np.size(inp), i + kernel_size + 1)
        region = inp[regionLB:regionUB]
        region = region.astype(np.float32)

        # Compute plane
        ax2 = second_bilateral_derivative[i] / 2 * ((np.arange(regionLB, regionUB) - i) ** 2)
        bx = first_bilateral_derivative[i] * (np.arange(regionLB, regionUB) - i) - ax2
        c = inp[i]

        quad_plane = ax2 + bx + c

        # quad_plane = (inp[i] + first_bilateral_derivative[i] * (np.arange(regionLB, regionUB) - i) +
        #             second_bilateral_derivative[i] * (np.arange(regionLB, regionUB) - i) ** 2)

        # I{delta}(x, vector)
        diff_from_plane = region - quad_plane

        # S-kernel
        s_kernel = gaussian_kernel_1d(sigma_intensity, np.abs(diff_from_plane))

        # TODO: Neighborhood inclusion kernel
        # inclusion = np.abs(gradient_vectors[regionLB:regionUB] - gradient_vectors[i]) <= R

        # Kernel
        kernel = spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (inp.shape[0] - i), (2 * kernel_size) + 1)] * s_kernel
        kernel /= np.sum(kernel)

        # Compute final value
        filtered_inp[i] = inp[i] + np.sum(diff_from_plane * kernel)

    return filtered_inp

if __name__ == '__main__':
    a = 10 # coefficient of x^2
    b = -5  # coefficient of x
    c = 0  # constant term

    # Define the range of x values
    x_values = np.linspace(-5, 5, 100).astype(float)  # Adjust the range and number of points as needed

    # Calculate the y values using the parabolic equation
    y_values = a * x_values ** 2 + b * x_values + c

    # Concatenate the two arrays
    inp_original = y_values
    inp_original = inp_original.clip(0, 255)

    inp = add_gauss_noise_1d(inp_original, 10)

    out = quadrateral_filter(inp, 4)
    out_bilat = bilateral_filter_1D(inp, 10, 50)
    out_trilat = trilateral_filter(inp, 3)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.plot(inp)
    plt.title('Original Input')




    plt.figure(figsize=(15, 5))
    plt.subplot(151)
    plt.plot(inp_original)
    plt.title('Original Input')

    plt.subplot(152)
    plt.plot(inp)
    plt.title('Noised Input')

    plt.subplot(153)
    plt.plot(out)
    plt.title('Quad Filter Output')

    plt.subplot(154)
    plt.plot(out_bilat)
    plt.title('Bilateral Filter Output')

    plt.subplot(155)
    plt.plot(out_trilat)
    plt.title('Trilateral Filter Output')

    plt.show()

    # Calculate average per pixel difference between the two outputs and original input
    error_quad = np.mean(np.abs(inp_original - out))
    error_bilateral = np.mean(np.abs(inp_original - out_bilat))
    print(f'Average per pixel difference between original and quad output: {error_quad}')
    print(f'Average per pixel difference between original and bilateral output: {error_bilateral}')


