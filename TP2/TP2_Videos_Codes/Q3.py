import cv2
import matplotlib.pyplot as plt
import numpy as np


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


def plot_correlation(corr):
    plt.figure(num=2, figsize=(4, 4))
    plt.clf()
    plt.rcParams["figure.figsize"] = (5, 5)
    plt.plot(corr, 'b', linewidth=0.5)
    plt.ylim([0, 1])
    plt.title("Correlation des histogrammes h et h-1")
    plt.xlabel("Numero de frames")
    plt.ylabel("Correlation (%)")
    plt.draw()
    plt.pause(0.0001)


def Calcule_Flow_histogramme(flow):
    bins = 64
    hist = cv2.calcHist([flow], [1, 0], None, [bins] * 2, [-bins, bins] * 2)

    # Elimination des valeurs statics (pas importantes dans ce cas)
    hist[hist[:, :] > np.std(hist) / 2] = np.std(hist) / 2

    # Normalisation non-lineaire
    hist[:, :] = (hist[:, :] / np.max(hist))

    # Histogramme avec la probabilité jointe
    plt.figure(num=1)
    plt.clf()
    plt.title("Histogramme 2D des composantes $V_x$ et $V_y$")
    plt.xlabel("Composante $V_x$")
    plt.ylabel("Composante $V_y$")
    plt.imshow(hist, interpolation='nearest')
    plt.colorbar()
    plt.draw()
    plt.pause(1e-3)

    return hist


# Ouverture du flux video
cap = cv2.VideoCapture("../TP2_Videos/Extrait3-Vertigo-Dream_Scene(320p).m4v")
# cap = cv2.VideoCapture(0)

_, frame1 = cap.read()  # Passe à l'image suivante
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # Passage en niveaux de gris

ret, frame2 = cap.read()
next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# Initialise les variables pour la comparations des histogrammes
flow = Find_flow(prvs, next)
hist_N = np.zeros_like(Calcule_Flow_histogramme(flow))
hist_N_moins1 = np.zeros_like(Calcule_Flow_histogramme(flow))
corr = []
index = 1

while ret:
    index += 1
    # Prendre le flow de l'image
    flow = Find_flow(prvs, next)

    # Compare les histogrammes
    hist_N_moins1 = hist_N
    hist_N = Calcule_Flow_histogramme(flow)
    corr.append(cv2.compareHist(hist_N_moins1, hist_N, cv2.HISTCMP_CORREL))

    # Calcule la correlation entre les imagems
    plot_correlation(corr)

    # Affiche la video
    cv2.imshow('Video', frame2)
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    prvs = next
    ret, frame2 = cap.read()
    if ret:  # Si il y a une image a être analysé
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

cap.release()
cv2.destroyAllWindows()
