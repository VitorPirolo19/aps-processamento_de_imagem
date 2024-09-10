import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path = ''  # imagem de um lago sei la
image = cv2.imread(img_path)

# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# gaussian blur para remover ruídos na imagem 
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# detector de bordas
edges = cv2.Canny(blurred, 50, 150)

_, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# desenha borda
contour_img = image.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 3)

# display
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(edges, cmap='gray'), plt.title('Edge Detection')
plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)), plt.title('Contours (Possible Garbage)')
plt.show()