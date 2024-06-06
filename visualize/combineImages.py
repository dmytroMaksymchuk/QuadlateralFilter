import numpy as np
import cv2 as cv

def combineImages(images):
    rows = len(images)
    cols = len(images[0])

    # All images are the same shape, combine with numpy
    imageRows = [np.hstack(images[i]) for i in range(rows)]
    return np.vstack(imageRows)

if __name__ == '__main__':
    spatials = [4, 6, 8, 10]
    intensities = [5, 15, 20, 30, 40]

    filter = "trilateral"

    images = [[cv.imread(f'../images/statue/noise_10/spatial_{spatial}/intensity_{intensity}/{filter}.jpg', cv.IMREAD_GRAYSCALE) for intensity in intensities] for spatial in spatials]

    combined = combineImages(images)

    cv.imwrite(f'../images/statue/noise_10/{filter}_combined.jpg', combined)