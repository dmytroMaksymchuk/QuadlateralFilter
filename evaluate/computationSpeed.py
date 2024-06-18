import math

import cv2 as cv
import time
import matplotlib.pyplot as plt

from bilateral.bilateralFilter import bilateral_filter
from quadlateral.quadlateralFilter2D import quadrateral_filter_2d
from trilateral.trilateralFilter2D import trilateral_filter_2d




if __name__ == '__main__':
    # test bilateral trilateral and quadlateral filter on speed with images

    path = '../images/speed/'
    images = ['statue_243x269.png', 'clouds_640x427.jpg', 'road_1600x1152.jpg']
    sigma_spatial = 3
    sigma_intensity = 30

    times = []

    for image in images:
        img = cv.imread(path + image, cv.IMREAD_GRAYSCALE)

        quad_start_time = time.time()
        quadrateral_filter_2d(img, sigma_spatial, sigma_intensity)
        quad_time_taken = time.time() - quad_start_time
        print(f'{image} - Quadlateral: {quad_time_taken}')

        bilat_start_time = time.time()
        start_time = time.time()
        bilateral_filter(img, sigma_intensity, sigma_spatial)
        bilat_time_taken = time.time() - start_time
        print(f'{image} - Bilateral: {bilat_time_taken}')

        trilat_start_time = time.time()
        trilateral_filter_2d(img, sigma_spatial, sigma_intensity)
        trilat_time_taken = time.time() - trilat_start_time
        print(f'{image} - Trilateral: {trilat_time_taken}')



        times.append((bilat_time_taken, trilat_time_taken, quad_time_taken))
        print(f'{image} - Bilateral: {bilat_time_taken}, Trilateral: {trilat_time_taken}, Quadlateral: {quad_time_taken}')


    print('Bilateral, Trilateral, Quadlateral')

