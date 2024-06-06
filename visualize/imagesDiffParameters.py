import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from evaluate.fastNoiseApprox import fastMeanNoiseEstimation
from helpers.gaussianHelper import add_gauss_noise_2d_image
from quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from trilateral.trilateralFilter2D import trilateral_filter_2d


if __name__ == '__main__':
    originalImage = cv.imread('../images/statue.png', cv.IMREAD_GRAYSCALE)
    noisedImage = add_gauss_noise_2d_image(originalImage, 10)

    sigmaSpatials = [4, 6, 8, 10]
    sigmaIntensities = [5, 15, 20, 30, 40]

    resultsSpatial = []
    for sigmaSpatial in sigmaSpatials:
        resultsIntensity = []
        for sigmaIntensity in sigmaIntensities:
            print("Spatial: ", sigmaSpatial, " Intensity: ", sigmaIntensity)

            bilateral = cv.bilateralFilter(noisedImage, sigmaSpatial*6+1, sigmaColor=sigmaIntensity, sigmaSpace=sigmaSpatial)
            tri = trilateral_filter_2d(noisedImage, sigma_spatial=sigmaSpatial, sigma_intensity=sigmaIntensity).clip(0, 255).astype(np.uint8)
            quad = quadrateral_filter_2d(noisedImage, sigma_spatial=sigmaSpatial, sigma_intensity=sigmaIntensity)[0].clip(0, 255).astype(np.uint8)

            noise_bilateral = fastMeanNoiseEstimation(originalImage, bilateral)
            noise_trilateral = fastMeanNoiseEstimation(originalImage, tri)
            noise_quad = fastMeanNoiseEstimation(originalImage, quad)

            resultsIntensity.append([noise_bilateral, noise_trilateral, noise_quad])

            path = '../images/statue/noise_10/spatial_' + str(sigmaSpatial) + '/intensity_' + str(sigmaIntensity) + '/'
            cv.imwrite(path + 'quadlateral.jpg', quad)
            cv.imwrite(path + 'bilateral.jpg', bilateral)
            cv.imwrite(path + 'trilateral.jpg', tri)
            cv.imwrite(path + 'noised.jpg', noisedImage)
            cv.imwrite(path + 'diff_quad.jpg', np.abs(quad.astype(np.float32) - originalImage).astype(np.uint8))
            cv.imwrite(path + 'diff_bilateral.jpg', np.abs(bilateral.astype(np.float32) - originalImage).astype(np.uint8))
            cv.imwrite(path + 'diff_trilateral.jpg', np.abs(tri.astype(np.float32) - originalImage).astype(np.uint8))

        resultsSpatial.append(resultsIntensity)

    print("Results:")
    print("Spatial: ", sigmaSpatials)
    print("Intensity: ", sigmaIntensities)
    print(resultsSpatial)

    # Prepare data for line plots
    bilateral_noise = np.array(
        [[resultsSpatial[i][j][0] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    trilateral_noise = np.array(
        [[resultsSpatial[i][j][1] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])
    quadlateral_noise = np.array(
        [[resultsSpatial[i][j][2] for j in range(len(sigmaIntensities))] for i in range(len(sigmaSpatials))])

    # save the results
    np.save('resultsStatue.npy', np.array(resultsSpatial))


    # Visualization with subplots for each sigma spatial for psnr
    fig, axes = plt.subplots(2, 2, figsize=(18, 6), sharey=True)

    for i, sigmaSpatial in enumerate(sigmaSpatials):
        ax = axes[i]
        ax.plot(sigmaIntensities, bilateral_noise[i], marker='o', label='Bilateral')
        ax.plot(sigmaIntensities, trilateral_noise[i], marker='s', label='Trilateral')
        ax.plot(sigmaIntensities, quadlateral_noise[i], marker='^', label='Quadlateral')

        ax.set_xlabel('Sigma Intensity')
        if i == 0:
            ax.set_ylabel('Estimated Noise Standard Deviation')
        ax.set_title(f'Spatial Sigma {sigmaSpatial}')
        ax.legend()
        ax.grid(True)

    fig.suptitle('Noise Estimation by Filter and Configuration')
    plt.show()



