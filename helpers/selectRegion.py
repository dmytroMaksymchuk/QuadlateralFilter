import cv2 as cv
import numpy as np

from helpers.increaseContrast import increase_contrast


def selectRegion(image, region):
    """
    Selects a region of interest from an image.
    :param image: The image from which to select the region.
    :param region: The region to select.
    :return: The selected region.
    """
    return image[region[0]:region[1], region[2]:region[3]]

if __name__ == '__main__':
    original = cv.imread('../images/statue.png', cv.IMREAD_GRAYSCALE)
    bilat = cv.imread('../images/statue/bilateral.jpg', cv.IMREAD_GRAYSCALE)
    trilateral = cv.imread('../images/statue/trilateral.jpg', cv.IMREAD_GRAYSCALE)
    quad = cv.imread('../images/statue/quadlateral.jpg', cv.IMREAD_GRAYSCALE)

    region = (100, 180, 110, 140)

    original_region = selectRegion(original, region)
    bilat_region = selectRegion(bilat, region)
    trilateral_region = selectRegion(trilateral, region)
    quad_region = selectRegion(quad, region)

    print("MSE Bilateral: ", np.mean(((original_region.astype(np.float32)) - bilat_region) ** 2))
    print("MSE Trilateral: ", np.mean(((original_region.astype(np.float32)) - trilateral_region) ** 2))
    print("MSE Quadlateral: ", np.mean(((original_region.astype(np.float32)) - quad_region) ** 2))

    diff_bilat = np.abs(bilat_region.astype(np.float32) - original_region)
    diff_trilateral = np.abs(trilateral_region.astype(np.float32) - original_region)
    diff_quad = np.abs(quad_region.astype(np.float32) - original_region)



    path = '../images/statue/'
    increase_contrast(path + 'diff_bilateral_region.jpg', path + 'diff_bilateral_contrast_region.jpg', 20.0)
    increase_contrast(path + 'diff_trilateral_region.jpg', path + 'diff_trilateral_contrast_region.jpg', 20.0)
    increase_contrast(path + 'diff_quad_region.jpg', path + 'diff_quad_contrast_region.jpg', 20.0)
    cv.waitKey(0)
