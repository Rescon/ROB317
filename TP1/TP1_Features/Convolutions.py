import numpy as np
import cv2

import matplotlib.pyplot as plt

# Lecture image en niveau de gris et conversion en float64
img = np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png', 0))
(h, w) = img.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes")

# Méthode directe
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
for y in range(1, h - 1):
    for x in range(1, w - 1):
        # Ixx
        # val = -2 * img[y, x] + img[y, x - 1] + img[y, x + 1]

        # Ixy
        # val = 1 * img[y - 1, x] - img[y, x - 1] - img[y + 1, x] + img[y, x + 1]

        # Iyy
        # val = -2 * img[y, x] + img[y - 1, x] + img[y + 1, x]

        val = 5 * img[y, x] - img[y - 1, x] - img[y, x - 1] - img[y + 1, x] - img[y, x + 1]
        img2[y, x] = min(max(val, 0), 255)
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print("Méthode directe :", time, "s")

plt.subplot(221)
plt.imshow(img2, cmap='gray')
plt.title('Convolution - Méthode Directe')

# Méthode filter2D
t1 = cv2.getTickCount()

# Ixx
# kernel = np.array([[1, -2, 1]])

# Ixy
# kernel = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]])

# Iyy
# kernel = np.array([[1], [-2], [1]])

kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
img3 = cv2.filter2D(img, -1, kernel)
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print("Méthode filter2D :", time, "s")

plt.subplot(222)
plt.imshow(img3, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('Convolution - filter2D')

plt.subplot(223)
plt.imshow(img, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('original image')

# la différence entre img2 et img3
plt.subplot(224)
plt.imshow(img2 - img3, cmap='gray', vmin=0.0, vmax=255.0)
plt.title('difference')

plt.show()
