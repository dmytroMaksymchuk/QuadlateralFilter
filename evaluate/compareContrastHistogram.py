import cv2 as cv
from matplotlib import pyplot as plt


def getHistogram(image):
    histogram = cv.calcHist([image], [0], None, [256], [0, 256])
    return histogram

if __name__ == '__main__':
    image = cv.imread('../images/statue.png', cv.IMREAD_GRAYSCALE)
    histogram_image = getHistogram(image)

    bilat = cv.imread('../images/statue/bilateral.jpg', cv.IMREAD_GRAYSCALE)
    histogram_bilat = getHistogram(bilat)

    trilateral = cv.imread('../images/statue/trilateral.jpg', cv.IMREAD_GRAYSCALE)
    histogram_trilateral = getHistogram(trilateral)

    quad = cv.imread('../images/statue/quadlateral.jpg', cv.IMREAD_GRAYSCALE)
    histogram_quad = getHistogram(quad)

    #plt.plot(histogram_image, color='black')
    #plt.plot(histogram_bilat, color='blue')
    plt.plot(histogram_trilateral, color='green')
    plt.plot(histogram_quad, color='red')
    plt.legend(['Original', 'Bilateral', 'Trilateral', 'Quadlateral'])
    plt.show()



