import math

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from trilateral.trilateralFilter2D import trilateral_filter_2d


def local_tonemap(hdr_image, s, sigma_spatial, sigma_intensity, filter='b'):
    # RGB to intensity/luminance
    intensity = 0.35 * hdr_image[:, :, 0] + 0.45 * hdr_image[:, :, 1] + 0.2 * hdr_image[:, :, 2]
    intensity = np.add(intensity, 0.01)*2
    color = (hdr_image.T / intensity.T).T


    if filter == 'q':
        base_layer = quadrateral_filter_2d(intensity, sigma_spatial, sigma_intensity)[0]
    elif filter == 'qu':
        base_layer = quadrateral_filter_2d(intensity, sigma_spatial, sigma_intensity, interpolation=True)[0]
    elif filter == 't':
        base_layer = trilateral_filter_2d(intensity, sigma_spatial, sigma_intensity)
    elif filter == 'b':
        base_layer = cv2.bilateralFilter(intensity, math.ceil(sigma_spatial*1.5)*2, sigma_intensity, sigma_spatial)
    else:
        raise ValueError('Invalid filter')


    detail_layer = intensity / base_layer
    #detail_layer = np.clip(detail_layer, 0, 10)

    large = np.where(detail_layer > 100)

    # Contrast reduction
    base_layer = np.sqrt(s * (base_layer - np.min(base_layer)) / (np.max(base_layer) - np.min(base_layer)))

    intensity = base_layer * detail_layer
    intensity = np.clip(intensity, 0, 1)

    plt.figure()
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=5)
    plt.imshow(detail_layer, cmap='bwr', norm=norm, interpolation='nearest')

    plt.colorbar()
    plt.title('Detail Layer')
    plt.savefig(f'images/resultImages/memorial/more/detail/{filter}_detail_layer.pdf')
    plt.show()
    np.save(f'images/resultImages/memorial/more/npy/{filter}_detail_layer.npy', detail_layer)

    tonemapped_image = (intensity.T * color.T).T

    return tonemapped_image

if __name__ == '__main__':
    s = 1000
    sigma_spatials = [8]
    sigma_intensity = [0.5]
    # sigma_spatials = [10]
    # sigma_intensity = [0.01]

    file = 'images/memorial_resized.hdr'
    hdr_image = cv2.imread(file, cv2.IMREAD_UNCHANGED)

    for spatial in sigma_spatials:
        for intensity in sigma_intensity:
            filter = 't'
            tonemapped_image = local_tonemap(hdr_image, s, spatial, intensity, filter=filter)
            cv2.imshow('Tonemapped Image', tonemapped_image)
            cv2.waitKey(0)
            cv2.imwrite(f"images/resultImages/memorial/more/try_{filter}_{s}_{spatial}_{intensity}.tif", tonemapped_image, [cv2.IMWRITE_TIFF_COMPRESSION, 0])


