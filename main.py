from PIL import Image, ImageFilter, ImageOps
import numpy as np
import matplotlib.pyplot as plt

img_path = '' 
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
plt.figure(figsize=(10, 10))

# original grayscale image
plt.subplot(2, 2, 1)
plt.imshow(img_np, cmap='gray')
plt.title('Original Grayscale Image')

# blurred image
plt.subplot(2, 2, 2)
plt.imshow(blurred_np, cmap='gray')
plt.title('Blurred Image')

# edge detection
plt.subplot(2, 2, 3)
plt.imshow(edges_np, cmap='gray')
plt.title('Edge Detection')

# thresholded (binary) and inverted image
plt.subplot(2, 2, 4)
plt.imshow(inverted_np, cmap='gray')
plt.title('Inverted Binary Image')

plt.show()