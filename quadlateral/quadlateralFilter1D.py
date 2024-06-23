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

def quadrateral_filter(inp, sigma_spatial, sigma_intensity, interpolation=False):
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
    k_vals = np.zeros_like(inp).astype(np.float32)
    uncert = np.zeros_like(inp).astype(np.float32)
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

        inclusion_grad = np.abs(first_bilateral_derivative[regionLB:regionUB] - first_bilateral_derivative[i]) <= R
        #inclusion_sec = np.abs(second_bilateral_derivative[regionLB:regionUB] - second_bilateral_derivative[i]) <= 0.02
        inclusion = inclusion_grad

        # Kernel
        kernel = spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (inp.shape[0] - i), (2 * kernel_size) + 1)] * s_kernel * inclusion
        k_vals[i] = np.sum(kernel)
        kernel /= np.sum(kernel)


        # Compute final value
        filtered_inp[i] = inp[i] + np.sum(diff_from_plane * kernel)

        if(i == 38):
            line = quad_plane

        if(interpolation):
            if i < 2:
                uncertainty = 0
            else:
                 uncertainty = 1/(1 + math.e ** ((2*((k_vals[i]) - np.mean(k_vals[:i]))/np.std(k_vals[:i])) + 1.5))

            uncert[i] = uncertainty

            intensity_diff = region - inp[i]
            bilat_range_kernel = gaussian_kernel_1d(sigma_intensity, np.abs(intensity_diff))
            bilat_kernel = bilat_range_kernel * spatial_kernel[max(kernel_size - i, 0):min(kernel_size + (inp.shape[0] - i), (2 * kernel_size) + 1)]
            bilat_kernel /= np.sum(bilat_kernel)
            bilat_pixel_value = np.sum(region * bilat_kernel)

            filtered_inp[i] = (1 - uncertainty) * filtered_inp[i] + uncertainty * bilat_pixel_value


    return filtered_inp, line, uncert

if __name__ == '__main__':

    # Define the range of x values
    num_points = 40
    padding = 10
    x_values = np.linspace(-7, 7, num_points).astype(np.float32)  # Adjust the range and number of points as needed

    # Calculate the y values using the parabolic equation
    padd = np.zeros(padding)
    # y_p1 = x_values[:num_points//4] * 0
    # y_p2 = x_values[num_points//4:num_points//2] * 2 + 1
    # y_p3 = (x_values[num_points//2:]-1) ** 6

    y_p1 = x_values * 0 + 100
    y_p2 = x_values * 0


    y_values = np.concatenate((y_p1, y_p2))

    # Concatenate the two arrays
    inp_original = y_values
    inp = add_gauss_noise_1d_signal(inp_original, 2)
    #inp = inp_original

    sigma_spatial = 4
    sigma_intensity = 10


    out5 = quadrateral_filter(inp, 4, 5)[0]
    out15 = quadrateral_filter(inp, 4, 15)[0]
    out25 = quadrateral_filter(inp, 4, 25)[0]
    out35_interp, trash, uncert = quadrateral_filter(inp, 4, 35, interpolation=True)
    out35, line, dsfds = quadrateral_filter(inp, 4, 35)

    bilat = bilateral_filter_1D(inp, 4, 35)
    trilat = trilateral_filter(inp, 4, 35)


    out5 = out5[padding:-padding]
    out15 = out15[padding:-padding]
    out25 = out25[padding:-padding]
    out35 = out35[padding:-padding]
    out35_interp = out35_interp[padding:-padding]
    uncert = uncert[padding:-padding]
    inp = inp[padding:-padding]
    inp_original = inp_original[padding:-padding]


    midpoint = 28
    fig, ax1 = plt.subplots()
    #ax2 = ax1.twinx()
    ax1.plot(inp, linewidth=1)
    ax1.plot(out35, linewidth=1)
    # ax1.plot(out35_interp, linewidth=1)
    # ax2.plot(uncert, linewidth=1, color='red')

    line = line[4:-4]
    x_line = np.linspace(midpoint - (sigma_spatial*1.5), midpoint + (sigma_spatial*1.5)+1, sigma_spatial*3+1)[4:-4]

    print(line)
    print(x_line)
    ax1.scatter(x_line, line.clip(-100,200), color='red')
    ax1.plot(x_line, line.clip(-100,200))

    ax1.legend(['input', 'quadrialteral', 'quadratic surface'], loc='center left')
    #ax2.legend(['uncertainty'], loc='center right')
    plt.title('Noisy Step Function')
    plt.savefig('../images/paper/step_funtion_problem.pdf')
    plt.show()



    # Calculate average per pixel difference between the two outputs and original input


