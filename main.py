import matplotlib.pyplot as plt
import cv2 as cv

from helpers.gaussianHelper import add_gauss_noise_2d
from bilateral.bilateralFilter import bilateral_filter

if __name__ == '__main__':

    img = cv.imread('images/image.jpg', cv.IMREAD_GRAYSCALE)
    img = add_gauss_noise_2d(img, 30)
    sigmaSpatial = 50
    sigmaIntensity = 50
    blured_img = bilateral_filter(img, sigmaSpatial, sigmaIntensity)
    blured_img_cv = cv.bilateralFilter(img, 10, sigmaSpatial, sigmaIntensity)

    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(blured_img, cmap='gray')
    plt.title('Bilateral Filter')
    plt.axis('off')


    plt.subplot(133)
    plt.imshow(blured_img_cv, cmap='gray')
    plt.title('Bilateral Filter OpenCV')
    plt.axis('off')



    plt.show()
