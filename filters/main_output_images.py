import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_trash_with_gaussian_blur(input_dir, output_dir):
    
    # Iterate through each image in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            
            # Step 2: Convert to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Step 3: Apply Gaussian blur to reduce noise
            blurred_image = cv2.GaussianBlur(gray_image, (25, 5), 0)
            
            # Step 4: Apply thresholding
            _, thresholded_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Save thresholded image
            output_path = os.path.join(output_dir, f"filtered_{filename}")
            cv2.imwrite(output_path, thresholded_image)
            
            print(f"Processed and saved: {output_path}")
            
    print("All images processed and saved.")

# Example usage
input_dir = r"../images/hard"  # Input directory with images
output_dir = r"../images/filtered_images"  # Output directory for processed images

detect_trash_with_gaussian_blur(input_dir, output_dir)
