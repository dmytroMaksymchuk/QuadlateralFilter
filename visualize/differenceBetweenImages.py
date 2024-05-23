import cv2 as cv
import numpy as np

if __name__ == '__main__':
    img1 = cv.imread('../images/golf2/diff_quad.jpg', cv.IMREAD_GRAYSCALE)
    img2 = cv.imread('../images/golf2/diff_bilateral.jpg', cv.IMREAD_GRAYSCALE)
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    diff = diff.clip(0, 255).astype(np.uint8)
    cv.imwrite('../images/golf2/diff_diff.jpg', diff)