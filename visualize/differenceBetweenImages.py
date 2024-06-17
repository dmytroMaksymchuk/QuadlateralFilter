import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from visualize.combineImages import combineImages

if __name__ == '__main__':
    img1 = cv.imread('../HDR/images/test/attrium_quad.png', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('../HDR/images/test/attrium_trilat.png', cv.IMREAD_GRAYSCALE)

    cv.imshow('img1', img1)
    cv.imshow('img2', img2)
    cv.waitKey(0)


    img1 = cv.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv.INTER_AREA)
    # img2 = combineImages([[img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2]])

    diff = img1.astype(np.float32) - img2.astype(np.float32)

    plt.figure(figsize=(10, 10))
    plt.imshow(diff, cmap='hot', vmin=-50, vmax=50)
    plt.colorbar()  # Show color scale
    plt.title('Difference between Quad and Trilat')
    plt.savefig('../images/statue/noise_10/diff_quad_noise_trilat_noise.jpg')
    plt.show()
