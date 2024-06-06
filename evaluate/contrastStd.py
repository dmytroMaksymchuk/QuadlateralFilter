import cv2
import numpy as np

def contrast_std(image):
    """
    Calculate the standard deviation of the image.
    :param image: The image to calculate the standard deviation of.
    :return: The standard deviation of the image.
    """
    return np.std(image)

if __name__ == '__main__':
    image = cv2.imread('../images/statue.png', cv2.IMREAD_GRAYSCALE)
    bilateral = cv2.imread('../images/statue/bilateral.jpg', cv2.IMREAD_GRAYSCALE)
    trilateral = cv2.imread('../images/statue/trilateral.jpg', cv2.IMREAD_GRAYSCALE)
    quadlateral = cv2.imread('../images/statue/quadlateral.jpg', cv2.IMREAD_GRAYSCALE)

    print("Contrast Original: ", contrast_std(image))
    print("Contrast Bilateral: ", contrast_std(bilateral))
    print("Contrast Trilateral: ", contrast_std(trilateral))
    print("Contrast Quadlateral: ", contrast_std(quadlateral))
    