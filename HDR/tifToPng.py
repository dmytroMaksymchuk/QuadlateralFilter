import cv2


def convert_tif_to_png(input_file_path, output_file_path):
    try:
        # Read the TIFF image
        img = cv2.imread(input_file_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"Could not open or find the image: {input_file_path}")

        # Save the image in PNG format
        cv2.imwrite(output_file_path, img)
        print(f"Successfully converted {input_file_path} to {output_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # Example usage
    input_file_path = 'images/resultImages/church_1/church_bilat_3_6_30.tif'
    output_file_path = 'images/resultImages/church_1/png/church_bilat_3_6_30.png'

    convert_tif_to_png(input_file_path, output_file_path)