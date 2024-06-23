import math

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from helpers.gaussianHelper import add_gauss_noise_2d_image
from quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from trilateral.trilateralFilter2D import trilateral_filter_2d
from skimage.metrics import structural_similarity as ssim

if __name__ == '__main__':
    img = cv.imread('../images/statue.png', cv.IMREAD_GRAYSCALE)

    sigmaSpatial = 10
    sigmaIntensity = 40

    kernelSize = math.ceil(sigmaSpatial * 1.5) * 2 + 1
    noised_img = add_gauss_noise_2d_image(img, 10)

    quad = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity, interpolation=True)[0]
    quad = quad.clip(0, 255).astype(np.uint8)

    trilateral = trilateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity)
    trilateral = trilateral.clip(0, 255).astype(np.uint8)

    bilateral = cv.bilateralFilter(noised_img.astype(np.uint8), kernelSize, sigmaIntensity, sigmaSpatial)
    gaussian = cv.GaussianBlur(noised_img, (kernelSize, kernelSize), sigmaSpatial, sigmaSpatial)


    print("Results:")
    print("Diff Quadlateral and Original: ", np.mean((quad.astype(np.float32) - img) ** 2))
    print("Diff Bilateral and Original: ", np.mean((bilateral.astype(np.float32) - img) ** 2))
    print("Diff Trilateral and Original: ", np.mean((trilateral.astype(np.float32) - img) ** 2))
    print("Diff Gaussian and Original: ", np.mean((gaussian.astype(np.float32) - img) ** 2))

    print("SSIM results:")
    print("Quadlateral: ", ssim(img, quad, data_range=quad.max() - quad.min()))
    print("Bilateral: ", ssim(img, bilateral, data_range=bilateral.max() - bilateral.min()))
    print("Trilateral: ", ssim(img, trilateral, data_range=trilateral.max() - trilateral.min()))
    print("Gaussian: ", ssim(img, gaussian, data_range=gaussian.max() - gaussian.min()))

    num, ssimQuad = ssim(img, quad, data_range=quad.max() - quad.min(), full=True)
    num2, ssimBilateral = ssim(img, bilateral, data_range=bilateral.max() - bilateral.min(), full=True)
    num3, ssimTrilateral = ssim(img, trilateral, data_range=trilateral.max() - trilateral.min(), full=True)

    ssimQuad *= 255
    ssimBilateral *= 255
    ssimTrilateral *= 255


    path = '../images/statue/garbage/'

    # cv.imwrite(path + 'quadlateral.jpg', quad.astype(np.uint8))
    # cv.imwrite(path + 'bilateral.jpg', bilateral)
    # cv.imwrite(path + 'trilateral.jpg', trilateral.astype(np.uint8))
    # cv.imwrite(path + 'gaussian.jpg', gaussian)
    # cv.imwrite(path + 'noised.jpg', noised_img.astype(np.uint8))
    # cv.imwrite(path + 'diff_quad.jpg', np.abs(quad.astype(np.float32) - img).astype(np.uint8))
    # cv.imwrite(path + 'diff_bilateral.jpg', np.abs(bilateral.astype(np.float32) - img).astype(np.uint8))
    # cv.imwrite(path + 'diff_trilateral.jpg', np.abs(trilateral.astype(np.float32) - img).astype(np.uint8))
    # cv.imwrite(path + 'diff_quad_bilateral.jpg', np.abs(quad.astype(np.float32) - bilateral).astype(np.uint8))
    # cv.imwrite(path + 'ssim_quad.jpg', ssimQuad.astype(np.uint8))
    # cv.imwrite(path + 'ssim_bilateral.jpg', ssimBilateral.astype(np.uint8))
    # cv.imwrite(path + 'ssim_trilateral.jpg', ssimTrilateral.astype(np.uint8))

    cv.waitKey(0)



