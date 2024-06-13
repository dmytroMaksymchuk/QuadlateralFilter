import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from visualize.combineImages import combineImages

if __name__ == '__main__':
    img1 = cv.imread('../images/statue/noise_10/diff_quad_noise.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('../images/statue/noise_10/diff_trilat_noise.jpg', cv.IMREAD_GRAYSCALE)


    img1 = img1[250:750, 250:750]
    img2 = img2[250:750, 250:750]
    # img2 = combineImages([[img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2]])

    diff = img1.astype(np.float32) - img2.astype(np.float32)

    plt.figure(figsize=(10, 10))
    plt.imshow(diff, cmap='hot', interpolation='nearest')
    plt.colorbar()  # Show color scale
    plt.title('Difference between Quad and Trilat')
    plt.savefig('../images/statue/noise_10/diff_quad_noise_trilat_noise.jpg')
    plt.show()
