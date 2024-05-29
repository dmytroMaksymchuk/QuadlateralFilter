from PIL import Image, ImageEnhance
import cv2 as cv
def increase_contrast(image_path, output_path, factor):
    # Open an image file
    with Image.open(image_path) as img:
        # Create an ImageEnhance object
        enhancer = ImageEnhance.Contrast(img)
        # Apply the contrast enhancement
        img_enhanced = enhancer.enhance(factor)
        # Save the enhanced image
        img_enhanced.save(output_path)

if __name__ == '__main__':
    path = '../images/clouds_8_10_10/'
    increase_contrast(path + 'diff_bilateral.jpg', path + 'diff_bilateral_contrast.jpg', 60.0)
    increase_contrast(path + 'diff_quad.jpg', path + 'diff_quad_contrast.jpg', 60.0)
    increase_contrast(path + 'diff_trilateral.jpg', path + 'diff_trilateral_contrast.jpg', 60.0)
