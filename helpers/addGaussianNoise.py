import cv2 as cv
import numpy as np


def add_gauss_noise_2d(img, sigma=1):
    # Compute image dimensions
    rows, cols = img.shape
    img = img.astype(np.float32)
    # Add Gaussian noise
    noise = np.random.normal(0, sigma, (rows, cols))

    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)

    return noisy_img

def add_gauss_noise_1d(inp, sigma=1):
    # Compute image dimensions
    inp = inp.astype(np.float32)
    # Add Gaussian noise
    noise = np.random.normal(0, sigma, np.size(inp))

    noisy_out = np.clip(inp + noise, 0, 255).astype(np.uint8)

    return noisy_out


if __name__ == '__main__':
    img = cv.imread('../images/image.jpg', cv.IMREAD_GRAYSCALE)

    noisy_img = add_gauss_noise(img, 10)
    cv.imshow('Noisy Image', noisy_img)
    cv.imshow('Original Image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
