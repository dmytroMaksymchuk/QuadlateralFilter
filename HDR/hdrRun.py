import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

from quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from trilateral.trilateralFilter2D import trilateral_filter_2d


def local_tonemap(hdr_image, s, sigma_spatial, sigma_intensity):
    # RGB to intensity/luminance
    intensity = 0.35 * hdr_image[:, :, 0] + 0.45 * hdr_image[:, :, 1] + 0.2 * hdr_image[:, :, 2]
    color = (hdr_image.T / intensity.T).T

    # REPLACE BY YOUR FILTER
    base_layer = quadrateral_filter_2d(intensity, sigma_spatial, sigma_intensity)
    # base_layer = cv2.bilateralFilter(intensity, math.ceil(sigma_spatial*1.5)*2, sigma_intensity, sigma_spatial)
    detail_layer = intensity / base_layer

    # Contrast reduction
    base_layer = np.sqrt(s * (base_layer - np.min(base_layer)) / (np.max(base_layer) - np.min(base_layer)))

    intensity = base_layer * detail_layer

    tonemapped_image = (intensity.T * color.T).T

    return tonemapped_image

if __name__ == '__main__':
    s = 3
    sigma_spatials = [3, 6, 10]
    sigma_intensity = [10, 30, 50]

    file = 'images/nancy_church_1_resized.hdr'
    hdr_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    for spatial in sigma_spatials:
        for intensity in sigma_intensity:
            tonemapped_image = local_tonemap(hdr_image, s, spatial, intensity)
            tonemapped_image = np.clip(tonemapped_image, 0, 1)
            cv2.imwrite(f"images/resultImages/church_1/church_quad_{s}_{spatial}_{intensity}.tif", tonemapped_image, [cv2.IMWRITE_TIFF_COMPRESSION, 0])


