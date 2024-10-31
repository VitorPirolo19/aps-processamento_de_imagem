from PIL import Image, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt

img_path = '/home/wirtz/projects/aps-processamento_de_imagem/images/hard/000551_jpg.rf.68c4b81ad94aa026cfa01279b0100e60.jpg' 
img = Image.open(img_path).convert('L')  # grayscale

# gaussian blur
blurred_img = img.filter(ImageFilter.GaussianBlur(radius=2))

# edge detection (using the FIND_EDGES filter)
edges_img = img.filter(ImageFilter.FIND_EDGES)

# convert to binary image using thresholding
threshold = 128
binary_img = img.point(lambda p: p > threshold and 255)

# invert the image 
inverted_img = ImageOps.invert(binary_img)

img_np = np.array(img)
blurred_np = np.array(blurred_img)
edges_np = np.array(edges_img)
binary_np = np.array(binary_img)
inverted_np = np.array(inverted_img)

# plotting the results using Matplotlib
# Save the figure to an image file instead of displaying it interactively
plt.figure(figsize=(10, 10))

# Original grayscale image
plt.subplot(2, 2, 1)
plt.imshow(img_np, cmap='gray')
plt.title('Original Grayscale Image')

# Blurred image
plt.subplot(2, 2, 2)
plt.imshow(blurred_np, cmap='gray')
plt.title('Blurred Image')

# Edge detection
plt.subplot(2, 2, 3)
plt.imshow(edges_np, cmap='gray')
plt.title('Edge Detection')

# Thresholded (binary) and inverted image
plt.subplot(2, 2, 4)
plt.imshow(inverted_np, cmap='gray')
plt.title('Inverted Binary Image')

# Save the plot to a file
plt.savefig('image_processing_results.png')