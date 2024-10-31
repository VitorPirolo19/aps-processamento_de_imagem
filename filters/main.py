import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

def detect_trash_with_gaussian_blur(image_path):
    # Step 1: Load the image
    image = cv2.imread(image_path)
    
    # Step 2: Convert to grayscale (easier for thresholding)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Apply Gaussian blur to reduce noise and smooth the image
    blurred_image = cv2.GaussianBlur(gray_image, (25, 5), 0)
    
    # Step 4: Apply thresholding
    _, thresholded_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Step 5: Plot each transformation step
    plt.figure(figsize=(15, 8))

    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    # Grayscale image
    plt.subplot(2, 2, 2)
    plt.imshow(gray_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    # Blurred image
    plt.subplot(2, 2, 3)
    plt.imshow(blurred_image, cmap='gray')
    plt.title('Gaussian Blurred Image')
    plt.axis('off')

    # Thresholded image (Trash Detection)
    plt.subplot(2, 2, 4)
    plt.imshow(thresholded_image, cmap='gray')
    plt.title('Thresholded Image (Trash Detection)')
    plt.axis('off')

    # Show the result
    plt.tight_layout()
    plt.show()


# Example usage
image_path = r"/home/rcorrei4/code/aps-processamento_de_imagem/images/easy/DJI_0040.JPG" # Use your image path
detect_trash_with_gaussian_blur(image_path)
