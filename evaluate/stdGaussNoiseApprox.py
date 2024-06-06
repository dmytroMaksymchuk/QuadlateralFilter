import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter
from scipy.stats import mode

from helpers.gaussianHelper import add_gauss_noise_2d_image


def calculate_local_std(image, window_size):
    padded_image = np.pad(image, window_size // 2, mode='reflect')
    std_values = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]
            std_values[i, j] = np.std(window)

    return std_values

if __name__ == '__main__':
    # Read the image
    image = cv2.imread('../images/statue.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    noise = cv2.imread('../images/statue/noised.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    bilateral = cv2.imread('../images/statue/bilateral.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    trilateral = cv2.imread('../images/statue/trilateral.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    quadlateral = cv2.imread('../images/statue/quadlateral.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    # Apply a sliding filter of [5, 5] to calculate the std
    window_size = 5
    original_std = calculate_local_std(image, window_size)
    noise_std = calculate_local_std(noise, window_size)
    bilateral_std = calculate_local_std(bilateral, window_size)
    trilateral_std = calculate_local_std(trilateral, window_size)
    quadlateral_std = calculate_local_std(quadlateral, window_size)

    # Display the images
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 3, 1)
    plt.imshow(original_std, cmap='hot', interpolation='nearest')
    plt.title('Original Image Local Std')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(noise_std, cmap='hot', interpolation='nearest')
    plt.title('Noised Image Local Std')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(np.abs(bilateral_std - original_std), cmap='hot', interpolation='nearest')
    plt.title('Bilateral Image Local Std')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(trilateral_std, cmap='hot', interpolation='nearest')
    plt.title('Trilateral Image Local Std')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(quadlateral_std, cmap='hot', interpolation='nearest')
    plt.title('Quadlateral Image Local Std')
    plt.axis('off')

    plt.show()

