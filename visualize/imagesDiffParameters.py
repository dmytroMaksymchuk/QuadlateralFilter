import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

from QuadlateralFilter.helpers.gaussianHelper import add_gauss_noise_2d_image
from QuadlateralFilter.quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from QuadlateralFilter.trilateral.trilateralFilter2D import trilateral_filter_2d


if __name__ == '__main__':
    originalImage = cv.imread('../images/clouds.png', cv.IMREAD_GRAYSCALE)
    noisedImage = add_gauss_noise_2d_image(originalImage, 10)
    cv2.imshow('Original', originalImage)
    cv2.imshow('Noised', noisedImage)

    sigmaSpatials = [3, 5, 8]
    sigmaIntensities = [5, 15, 25, 35]

    resultsSpatial = []
    resultsSSIM = []
    for sigmaSpatial in sigmaSpatials:
        resultsIntensity = []
        resultsSSIMIntensity = []
        for sigmaIntensity in sigmaIntensities:
            print("Spatial: ", sigmaSpatial, " Intensity: ", sigmaIntensity)

            quad, trash, trash2 = quadrateral_filter_2d(noisedImage, sigma_spatial=sigmaSpatial, sigma_intensity=sigmaIntensity, interpolation=True)
            quad = quad.clip(0, 255).astype(np.uint8)
            bilateral = cv.bilateralFilter(noisedImage, sigmaSpatial*6+1, sigmaColor=sigmaIntensity, sigmaSpace=sigmaSpatial)
            tri = trilateral_filter_2d(noisedImage, sigma_spatial=sigmaSpatial, sigma_intensity=sigmaIntensity).clip(0, 255).astype(np.uint8)
            gaussian = cv.GaussianBlur(noisedImage, (sigmaSpatial*6+1, sigmaSpatial*6+1), sigmaSpatial, sigmaSpatial)

            psnr_quad = cv.PSNR(originalImage, quad)
            psnr_bilateral = cv.PSNR(originalImage, bilateral)
            psnr_trilateral = cv.PSNR(originalImage, tri)
            psnr_gaussian = cv.PSNR(originalImage, gaussian)

            mssim_quad = ssim(originalImage, quad, data_range=quad.max() - quad.min())
            mssim_bilateral = ssim(originalImage, bilateral, data_range=bilateral.max() - bilateral.min())
            mssim_trilateral = ssim(originalImage, tri, data_range=tri.max() - tri.min())
            mssim_gaussian = ssim(originalImage, gaussian, data_range=gaussian.max() - gaussian.min())

            resultsIntensity.append([psnr_gaussian, psnr_bilateral, psnr_trilateral, psnr_quad])
            resultsSSIMIntensity.append([mssim_gaussian, mssim_bilateral, mssim_trilateral, mssim_quad])

        resultsSpatial.append(resultsIntensity)
        resultsSSIM.append(resultsSSIMIntensity)

    print("Results:")
    print("Spatial: ", sigmaSpatials)
    print("Intensity: ", sigmaIntensities)
    print(resultsSpatial)

    # Prepare data for line plots
    gaussian_psnr = np.array(
        [[resultsSpatial[i][j][0] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    bilateral_psnr = np.array(
        [[resultsSpatial[i][j][1] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    trilateral_psnr = np.array(
        [[resultsSpatial[i][j][2] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    quadlateral_psnr = np.array(
        [[resultsSpatial[i][j][3] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])


    # Visualization with subplots for each sigma spatial for psnr
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, sigmaSpatial in enumerate(sigmaSpatials):
        ax = axes[i]
        ax.plot(sigmaIntensities, gaussian_psnr[i], marker='x', label='Gaussian')
        ax.plot(sigmaIntensities, bilateral_psnr[i], marker='o', label='Bilateral')
        ax.plot(sigmaIntensities, trilateral_psnr[i], marker='s', label='Trilateral')
        ax.plot(sigmaIntensities, quadlateral_psnr[i], marker='^', label='Quadlateral')

        ax.set_xlabel('Sigma Intensity')
        if i == 0:
            ax.set_ylabel('PSNR (dB)')
        ax.set_title(f'Spatial Sigma {sigmaSpatial}')
        ax.legend()
        ax.grid(True)

    fig.suptitle('PSNR by Filter and Configuration')
    plt.show()

    # Prepare data for line plots SSIM
    gaussian_ssim = np.array(
        [[resultsSSIM[i][j][0] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    bilateral_ssim = np.array(
        [[resultsSSIM[i][j][1] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    trilateral_ssim = np.array(
        [[resultsSSIM[i][j][2] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    quadlateral_ssim = np.array(
        [[resultsSSIM[i][j][3] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])

    # Visualization with subplots for each sigma spatial for ssim
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for i, sigmaSpatial in enumerate(sigmaSpatials):
        ax = axes[i]
        ax.plot(sigmaIntensities, gaussian_ssim[i], marker='x', label='Gaussian')
        ax.plot(sigmaIntensities, bilateral_ssim[i], marker='o', label='Bilateral')
        ax.plot(sigmaIntensities, trilateral_ssim[i], marker='s', label='Trilateral')
        ax.plot(sigmaIntensities, quadlateral_ssim[i], marker='^', label='Quadlateral')

        ax.set_xlabel('Sigma Intensity')
        if i == 0:
            ax.set_ylabel('SSIM')
        ax.set_title(f'Spatial Sigma {sigmaSpatial}')
        ax.legend()
        ax.grid(True)

    fig.suptitle('SSIM by Filter and Configuration')
    plt.show()








