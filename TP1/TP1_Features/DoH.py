import numpy as np
import cv2

from matplotlib import pyplot as plt

# Lecture image en niveau de gris et conversion en float64
img = np.float64(cv2.imread('../Image_Pairs/Graffiti0.png', cv2.IMREAD_GRAYSCALE))
(h, w) = img.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes")
print("Type de l'image :", img.dtype)

# Début du calcul
t1 = cv2.getTickCount()
Theta = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
# Mettre ici le calcul du déterminant de la matrice hessienne
kernel_xx = np.array([[1, -2, 1]])
kernel_yy = np.array([[1], [-2], [1]])
kernel_x = np.array([[-1, 0, 1]])
kernel_y = np.array([[1], [0], [-1]])

Ixx = cv2.filter2D(Theta, -1, kernel_xx)
Iyy = cv2.filter2D(Theta, -1, kernel_yy)
Ix = cv2.filter2D(Theta, -1, kernel_x)
Ixy = cv2.filter2D(Ix, -1, kernel_y)

Theta = Ixx*Iyy - Ixy**2
# Calcul des maxima locaux et seuillage
Theta_maxloc = cv2.copyMakeBorder(Theta, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
d_maxloc = 3
seuil_relatif = 0.01
se = np.ones((d_maxloc, d_maxloc), np.uint8)
Theta_dil = cv2.dilate(Theta, se)
# Suppression des non-maxima-locaux
Theta_maxloc[Theta < Theta_dil] = 0.0
# On néglige également les valeurs trop faibles
Theta_maxloc[Theta < seuil_relatif * Theta.max()] = 0.0
t2 = cv2.getTickCount()
time = (t2 - t1) / cv2.getTickFrequency()
print("Mon calcul de la fonction d'intérêt :", time, "s")
print("Nombre de cycles par pixel :", (t2 - t1) / (h * w), "cpp")

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title('Image originale')

plt.subplot(132)
plt.imshow(Theta, cmap='gray')
plt.title('Déterminant de la hessienne')

# Création d'un cercle pour indiquer les points d'intérêt
rayon_cercle = 11
se_disk1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rayon_cercle, rayon_cercle))
se_disk2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rayon_cercle - 2, rayon_cercle - 2))
se_disk1bis = cv2.copyMakeBorder(se_disk2, 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, value=0)
se_cercle = se_disk1 - se_disk1bis
# Dilatation : un point se transforme en un cercle
Theta_ml_dil = cv2.dilate(Theta_maxloc, se_cercle)
# Relecture image pour affichage couleur
Img_pts = cv2.imread('../Image_Pairs/Graffiti0.png', cv2.IMREAD_COLOR)
(h, w, c) = Img_pts.shape
print("Dimension de l'image :", h, "lignes x", w, "colonnes x", c, "canaux")
print("Type de l'image :", Img_pts.dtype)
# On affiche les points (cercles) en rouge
Img_pts[Theta_ml_dil > 0] = [255, 0, 0]
plt.subplot(133)
plt.imshow(Img_pts)
plt.title('Points saillants')

plt.show()
