import math

import numpy as np
import cv2
from matplotlib import pyplot as plt

from helpers.gaussianHelper import add_gauss_noise_2d_image

def filter2D(image, kernel):
    # Get the dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the amount of padding needed for each dimension
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Create an output image of the same size as the original
    filtered_image = np.zeros_like(image)

    # Perform the convolution operation
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Apply the kernel (element-wise multiplication and sum)
            filtered_value = np.sum(roi * kernel)

            # Assign the computed value to the output image
            filtered_image[i, j] = filtered_value

    return filtered_image


def fastMeanNoiseEstimation(original_image, image):
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
    convolved_original = cv2.filter2D(original_image, -1, kernel)
    convolved_image = cv2.filter2D(image, -1, kernel)
    difference = np.clip(convolved_image.astype(np.float32) - convolved_original, 0, 255).astype(np.uint8)

    cv2.imshow('conv', convolved_original)
    cv2.imshow('convolver_filter', convolved_image)

    cv2.waitKey(0)

    plt.figure(figsize=(10, 10))
    plt.imshow(difference, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Show color scale
    plt.title('convolved_original')
    plt.show()

    sigma_noise = np.sqrt(np.pi / 2) * 1 / (6 * (image.shape[0]-2) * (image.shape[1] - 2)) * np.sum(np.abs(difference))
    return np.mean(difference)

if __name__ == '__main__':
    original_image = cv2.imread('../images/statue.png', cv2.IMREAD_GRAYSCALE)
    #image = add_gauss_noise_2d_image(original_image, 9)
    image = cv2.imread('../images/statue/noise_10/spatial_8/intensity_30/bilateral.jpg', cv2.IMREAD_GRAYSCALE)

    sigma = fastMeanNoiseEstimation(original_image, image)
    print("Estimated Gaussian Noise Standard Deviation: ", sigma)

