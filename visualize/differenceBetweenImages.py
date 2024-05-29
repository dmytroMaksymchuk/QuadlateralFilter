import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img1 = cv.imread('../images/clouds_8_25_10_incl/quadInclud.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('../images/clouds_8_25_10_incl/quadlateral.jpg', cv.IMREAD_GRAYSCALE)
    diff = img1.astype(np.float32) - img2.astype(np.float32)

    plt.figure(figsize=(10, 10))
    plt.imshow(diff)
    plt.colorbar()  # Show color scale
    plt.title('Difference between Quadlateral and Noised')
    plt.show()

    diff = diff.clip(0, 255).astype(np.uint8)
    cv.imwrite('../images/golf2/quadMinusNoised.jpg', diff)