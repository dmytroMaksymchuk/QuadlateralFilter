import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from helpers.gaussianHelper import add_gauss_noise_2d_image
from quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from trilateral.trilateralFilter2D import trilateral_filter_2d
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    img = cv.imread('../images/clouds.png', cv.IMREAD_GRAYSCALE)

    sigmaSpatial = 8
    sigmaIntensity = 25

    kernelSize = sigmaSpatial * 6 + 1
    noised_img = add_gauss_noise_2d_image(img, 10)

    quad = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity)[0]
    quad = quad.clip(0, 255).astype(np.uint8)

    quadInclud01 = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity, inclusion_threshold=0.01)[0]
    quadInclud01 = quadInclud01.clip(0, 255).astype(np.uint8)

    quadInclud03 = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity, inclusion_threshold=0.03)[0]
    quadInclud03 = quadInclud03.clip(0, 255).astype(np.uint8)

    quadInclud05 = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity, inclusion_threshold=0.05)[0]
    quadInclud05 = quadInclud05.clip(0, 255).astype(np.uint8)

    quadInclud1 = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity, inclusion_threshold=0.1)[0]
    quadInclud1 = quadInclud1.clip(0, 255).astype(np.uint8)


    path = '../images/clouds_8_25_10_incl/'

    cv.imwrite(path + 'quadlateral.jpg', quad.astype(np.uint8))
    cv.imwrite(path + 'quadInclud01.jpg', quadInclud01)
    cv.imwrite(path + 'quadInclud03.jpg', quadInclud03)
    cv.imwrite(path + 'quadInclud05.jpg', quadInclud05)
    cv.imwrite(path + 'quadInclud1.jpg', quadInclud1)

    cv.waitKey(0)



