import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def contrast_std(image):
    """
    Calculate the standard deviation of the image.
    :param image: The image to calculate the standard deviation of.
    :return: The standard deviation of the image.
    """
    return np.std(image)

if __name__ == '__main__':
    image = cv.imread('../images/statue.png', cv.IMREAD_GRAYSCALE)

    spatials = [4, 6, 8, 10]
    intensities = [5, 15, 20, 30, 40]

    images = [[cv.imread(f'../images/statue/noise_10/spatial_{spatial}/intensity_{intensity}/{filter}.jpg',
                         cv.IMREAD_GRAYSCALE) for intensity in intensities] for spatial in spatials]

    resultSpatial = []
    for sigmaSpatial in spatials:
        resultIntensity = []
        for sigmaIntensity in intensities:
            quadlateral = cv.imread(f'../images/statue/noise_10/spatial_{sigmaSpatial}/intensity_{sigmaIntensity}/quadlateral.jpg', cv.IMREAD_GRAYSCALE)
            trilateral = cv.imread(f'../images/statue/noise_10/spatial_{sigmaSpatial}/intensity_{sigmaIntensity}/trilateral.jpg', cv.IMREAD_GRAYSCALE)
            print("Spatial: ", sigmaSpatial, " Intensity: ", sigmaIntensity)

            contrastTri = contrast_std(trilateral)
            contrastQuad = contrast_std(quadlateral)
            resultIntensity.append([contrastTri, contrastQuad])
        resultSpatial.append(resultIntensity)


    # Visualize
    trilateral_contrast = np.array(
        [[resultSpatial[i][j][0] for j in range(len(intensities))] for i in range(len(spatials))])
    quadlateral_contrast = np.array(
        [[resultSpatial[i][j][1] for j in range(len(intensities))] for i in range(len(spatials))])

    # Visualization with subplots for each sigma spatial for noise_estimation

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=True)

    for i, sigmaSpatial in enumerate(spatials):
        row = i // 2
        col = i % 2
        ax = axes[row][col]
        ax.plot(intensities, trilateral_contrast[i], label='Trilateral')
        ax.plot(intensities, quadlateral_contrast[i], label='Quadrilateral')
        ax.set_title('Contrast for Sigma Spatial: ' + str(sigmaSpatial))
        ax.set_xlabel('Sigma Intensity')
        ax.set_ylabel('Contrast')
        ax.legend()
        ax.grid(True)
    fig.savefig(f'../images/paper/contrast_std.pdf')
    plt.show()

    