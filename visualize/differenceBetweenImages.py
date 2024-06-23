import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from visualize.combineImages import combineImages

if __name__ == '__main__':
    img1 = cv.imread('../HDR/images/resultImages/memorial/more/mem_quad_uncert_20_8_0.02.tif',cv.IMREAD_UNCHANGED)
    img2 = cv.imread('../HDR/images/resultImages/memorial/more/memorial_trilat_uncert_20_8_0.02.tif', cv.IMREAD_UNCHANGED)


    #img1 = cv.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv.INTER_AREA)
    # img2 = combineImages([[img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2], [img2, img2, img2, img2, img2]])

    diff = img1.astype(np.float32) - img2.astype(np.float32)

    plt.figure(figsize=(12, 6))
    plt.imshow(diff, cmap='bwr', vmin=-10, vmax=10, interpolation='nearest')
    plt.colorbar()
    plt.title('Difference with Interpolation')
    #plt.savefig('../images/paper/to_review/quad_trilateral_diff.pdf')
    plt.show()