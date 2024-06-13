import cv2 as cv
def resizeImage(image, width, height):
    new_width = int(image.shape[1] * width)
    new_height = int(image.shape[0] * height)
    resized_hdr_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return resized_hdr_image


if __name__ == '__main__':
    image = cv.imread('images/forest_path.hdr', cv.IMREAD_UNCHANGED)
    resized_image = resizeImage(image, 0.3, 0.3)
    print(resized_image.shape)
    cv.imwrite('images/forest_path_resized.hdr', resized_image, [cv.IMWRITE_HDR_COMPRESSION_RLE, 0])