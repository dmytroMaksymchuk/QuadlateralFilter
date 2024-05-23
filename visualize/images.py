import cv2 as cv
import numpy as np

from QuadlateralFilter.helpers.gaussianHelper import add_gauss_noise_2d_image
from QuadlateralFilter.quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from QuadlateralFilter.trilateral.trilateralFilter2D import trilateral_filter_2d

if __name__ == '__main__':
    img = cv.imread('../images/golf_low_contrast.jpg', cv.IMREAD_GRAYSCALE)
    sigmaSpatial = 3
    sigmaIntensity = 10
    noised_img = add_gauss_noise_2d_image(img, 10)

    quad, trash = quadrateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity)
    quad = quad.clip(0, 255).astype(np.uint8)

    trilateral = trilateral_filter_2d(noised_img, sigmaSpatial, sigmaIntensity)
    trilateral = trilateral.clip(0, 255).astype(np.uint8)

    bilateral = cv.bilateralFilter(noised_img.astype(np.uint8), 19, sigmaIntensity, sigmaSpatial)


    print("Results:")
    print("Diff Quadlateral and Original: ", np.mean(np.abs(quad.astype(np.float32) - img)))
    print("Diff Bilateral and Original: ", np.mean(np.abs(bilateral.astype(np.float32) - img)))
    print("Diff Trilateral and Original: ", np.mean(np.abs(trilateral.astype(np.float32) - img)))

    cv.imwrite('../images/golf2/quadlateral.jpg', quad.astype(np.uint8))
    cv.imwrite('../images/golf2/bilateral.jpg', bilateral)
    cv.imwrite('../images/golf2/trilateral.jpg', trilateral.astype(np.uint8))
    cv.imwrite('../images/golf2/noised.jpg', noised_img.astype(np.uint8))
    cv.imwrite('../images/golf2/diff_quad.jpg', np.abs(quad.astype(np.float32) - img).astype(np.uint8))
    cv.imwrite('../images/golf2/diff_bilateral.jpg', np.abs(bilateral.astype(np.float32) - img).astype(np.uint8))
    cv.imwrite('../images/golf2/diff_trilateral.jpg', np.abs(trilateral.astype(np.float32) - img).astype(np.uint8))
    cv.imwrite('../images/golf2/diff_quad_bilateral.jpg', np.abs(quad.astype(np.float32) - bilateral).astype(np.uint8))

    cv.imshow('Original', img)
    cv.imshow('Noised', noised_img.astype(np.uint8))
    cv.imshow('Quadlateral', quad)
    cv.imshow('Bilateral', bilateral)
    cv.imshow('Trilateral', trilateral)
    cv.imshow('Diff Quadlateral and Original', np.abs(quad.astype(np.float32) - img).astype(np.uint8))
    cv.imshow('Diff Bilateral and Original', np.abs(bilateral.astype(np.float32) - img).astype(np.uint8))
    cv.imshow('Diff Trilateral and Original', np.abs(trilateral.astype(np.float32) - img).astype(np.uint8))
    cv.imshow('Diff Quadlateral and Bilateral', np.abs(quad.astype(np.float32) - bilateral).astype(np.uint8))
    cv.waitKey(0)



