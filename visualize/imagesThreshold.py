import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from helpers.gaussianHelper import add_gauss_noise_2d_image
from quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from trilateral.trilateralFilter2D import trilateral_filter_2d
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    noised_img = cv.imread('../images/statue/noise_10/spatial_8/intensity_30/noised.jpg', cv.IMREAD_GRAYSCALE)

    sigmaSpatial = 8
    sigmaIntensity = 30

    kernelSize = math.ceil(sigmaSpatial * 1.5) * 2 + 1

    quad = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity)[0]
    quad = quad.clip(0, 255).astype(np.uint8)

    quad_interp, uncert, trash = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity, interpolation=True)
    quad_interp = quad_interp.clip(0, 255).astype(np.uint8)

    cv.imwrite('../images/statue/uncertainty/paper/noised.jpg', noised_img)
    cv.imwrite('../images/statue/uncertainty/paper/filteredInterp.jpg', quad_interp)
    cv.imwrite('../images/statue/uncertainty/paper/filtered.jpg', quad)





