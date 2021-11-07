import cv2
import numpy as np
import skimage
from skimage.feature import hog
from numpy import shape


def Calcule_2D_YUV_histogramme(ret, frame):
    bins = 64
    if ret:
        # Read image & transform in YUV
        yuv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # Color_map for the YUV
        color_uv = np.zeros((bins, bins, 3), np.uint8)
        u, v = np.indices(color_uv.shape[:2])
        color_uv[:, :, 0] = 50
        color_uv[:, :, 1] = u * 256 / bins
        color_uv[:, :, 2] = v * 256 / bins
        color_uv = cv2.cvtColor(color_uv, cv2.COLOR_YUV2BGR)

        # 2D Histogram
        hist_norm = cv2.calcHist([yuv_image], [1, 2], None, [bins] * 2, [-0, 256] * 2)

        # Normalisation non-lineaire/ lineaire
        hist_norm[:, :] = np.log(hist_norm[:, :])
        hist_norm = np.clip(hist_norm, 0, np.max(hist_norm))
        hist_norm[:, :] = (hist_norm[:, :] / np.max(hist_norm))

        return hist_norm


def show_video(frame):
    # Display the images
    cv2.imshow('Video originale', frame)


def Find_flow(prvs, next):
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None,
                                        pyr_scale=0.5,  # Taux de réduction pyramidal
                                        levels=3,  # Nombre de niveaux de la pyramide
                                        winsize=15,
                                        # Taille de fenÃªtre de lissage (moyenne) des coefficients polynomiaux
                                        iterations=3,  # Nb d'itération par niveau
                                        poly_n=7,  # Taille voisinage pour approximation polynomiale
                                        poly_sigma=1.5,  # E-T Gaussienne pour calcul dérivés
                                        flags=0)
    return flow


def Calcule_Flow_histogramme(flow):
    bins = 64
    hist = cv2.calcHist([flow], [0, 1], None, [bins] * 2, [-bins, bins] * 2)

    # Elimination des valeurs statics (pas importantes dans ce cas)
    hist[hist[:, :] > np.std(hist) / 2] = np.mean(hist)

    # Normalisation non-lineaire
    hist[:, :] = (hist[:, :]) ** 0.5
    hist[:, :] = (hist[:, :] / np.max(hist))
    return hist


def apply_hog(image, number_ori):
    fd, hog_image = hog(image, orientations=number_ori, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True)  # , multichannel=True)
    number_histograms = (int(shape(image)[0] / 16) * int(shape(image)[1] / 16))
    teste = np.reshape(np.array(fd), (number_histograms, number_ori))
    histogram_hog = np.sum(teste, axis=0)
    histogram_hog = np.reshape(histogram_hog, (number_ori))
    histogram_hog[:] = (histogram_hog[:] / np.max(histogram_hog)) * 256
    return histogram_hog


def eqm_images(image1, image2):
    # 'Erreur quadratique moyenne' entre les images
    erreur = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    erreur /= float(image1.shape[0] * image1.shape[1])
    return erreur


def compare_images(image1, image2):
    m = eqm_images(image1, image2)
    # s: Compute the mean structural similarity index between two images
    s = skimage.measure.compare_ssim(image1, image2, multichannel=True)
    return m * s


# Reads the video from file
capture = cv2.VideoCapture("../TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")
# Reads video from webcam
# capture = cv2.VideoCapture(0)

_, frame1 = capture.read()  # Passe à l'image suivante
ret, frame2 = capture.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Passage en niveaux de gris
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Initialise les variables pour la comparations des histogrammes
yuvHist = np.zeros_like(Calcule_2D_YUV_histogramme(ret, frame2))
yuvHist_old = np.zeros_like(Calcule_2D_YUV_histogramme(ret, frame2))

# Quantité de orientations pour le HOG
number_ori = 8

# Initialise les variables pour la comparations des histogrammes
flow = Find_flow(prvs, next)
VxVyHist = np.zeros_like(Calcule_Flow_histogramme(flow))
VxVyHist_old = np.zeros_like(Calcule_Flow_histogramme(flow))
hog_hist = apply_hog(frame2, number_ori)

# Counter pour le frames
countFrames = 0

# Variables utilisées pour la détéction de plan
plans = []
plans_count = 0

while True:
    if not ret:
        break
    # Affiche la video
    show_video(frame2)

    detec_yuv = 0
    detec_flow = 0
    detec_hog = 0

    """
	@ METHODE 1: YUV Histogramme
	@ Threshold: 0.85
	"""
    # Sauvegard l'ancien et le nouveau histogramme
    yuvHist_old = yuvHist
    yuvHist = Calcule_2D_YUV_histogramme(ret, frame2)
    yuv_toCorr = cv2.compareHist(yuvHist_old, yuvHist, cv2.HISTCMP_CORREL)
    if yuv_toCorr < 0.85:
        detec_yuv = 1

    """
	@ METHODE 2: (Vx,Vy)Histogramme
	Seuil: 0.18
	"""
    # Calcule le flow de l'image
    flow = Find_flow(prvs, next)
    # Compare les histogrammes
    VxVyHist_old = VxVyHist
    VxVyHist = Calcule_Flow_histogramme(flow)
    flow_toCorr = cv2.compareHist(VxVyHist_old, VxVyHist, cv2.HISTCMP_CORREL)
    if flow_toCorr < 0.18:
        detec_flow = 1

    """
	@ METHODE 3: HOG
	Seuil: 0.95
	"""
    # Calcule le HOG
    hog_hist_old = hog_hist
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hog_hist = apply_hog(frame2, number_ori)
    hog_hist_toCorr = np.corrcoef(hog_hist_old, hog_hist)[0, 1]
    if hog_hist_toCorr < 0.95:
        detec_hog = 1

    # Ponderation pour détécter le changement de plan
    alpha = 1
    beta = 1
    gamma = 1
    detec_combine = alpha * detec_yuv + beta * detec_flow + gamma * detec_hog

    # Vote majoritaire pour la détéction de plans
    if detec_combine > 1:
        if plans_count == 0:
            plans.append([0, countFrames])
            print("Plan détécté entre les frames: ", plans[plans_count])
            plans_count += 1
        else:
            plans.append([plans[plans_count - 1][1] + 1, countFrames])
            print("Plan détécté entre les frames: ", plans[plans_count])
            plans_count += 1
    countFrames += 1

    # Type "q" to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prvs = next
    ret, frame2 = capture.read()
    if ret:
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

cv2.destroyAllWindows()
