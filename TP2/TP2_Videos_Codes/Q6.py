import cv2
import numpy as np
import skimage
from numpy import shape
from skimage.feature import hog


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
    hist = cv2.calcHist([flow], [1, 0], None, [bins] * 2, [-bins, bins] * 2)

    # Elimination des valeurs statics (pas importantes dans ce cas)
    # hist[hist[:,:]>np.std(hist)/2] = np.mean(hist)
    hist[hist[:, :] > np.std(hist) / 2] = np.std(hist) / 2

    # Normalisation non-lineaire
    # hist[:,:] = (hist[:,:])**0.5
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


def flow_hist(flow, step):
    mag_total, ang_total = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])  # Conversion cartÃ©sien vers polaire
    ang_total = (ang_total * 180) / (np.pi)  # Teinte (codÃ©e sur [0..179] dans OpenCV) <--> Argument

    ang_total = np.reshape(ang_total, shape(ang_total)[0] * shape(ang_total)[1])
    mag_total = np.reshape(mag_total, shape(mag_total)[0] * shape(mag_total)[1])

    angle = 0
    # step=15
    step_angle = 360 / step
    flow_mag_hist = []
    flow_angle_hist = []
    flow_hist_descript = []
    for i in range(step):
        angle_aux = ang_total[ang_total[:] >= angle]
        mag_aux = mag_total[ang_total[:] >= angle]
        if i == step - 1:
            angle_int = angle_aux[angle_aux[:] <= (angle + step_angle)]
            mag_int = mag_aux[angle_aux[:] <= (angle + step_angle)]
        else:
            angle_int = angle_aux[angle_aux[:] < (angle + step_angle)]
            mag_int = mag_aux[angle_aux[:] < (angle + step_angle)]

        flow_mag_hist.append(np.sum(mag_int))
        flow_angle_hist.append(shape(angle_int)[0])

        # print(angle, angle+step_angle,"---",mag_sum, shape(mag_total)[0], "//", shape(angle_int)[0], shape(ang_total)[0])#, ":", angle_aux_sum)

        # hist_flow_ori.append([mag_total, ang_total])

        angle += step_angle
    flow_hist_descript.append(flow_mag_hist)
    flow_hist_descript.append(flow_angle_hist)
    return flow_hist_descript


def Classification_type_plan(img, compare_angMag):
    img[img[:, :] == 1] = 0
    mean_img = np.mean(img)
    std_img = np.std(img)
    ratio_img = std_img / mean_img

    sum_H = []
    H0 = img[0:32, 0:32]
    H1 = img[0:32, 32:64]
    H2 = img[32:64, 0:32]
    H3 = img[32:64, 32:64]
    sum_H.append(np.sum(H0))
    sum_H.append(np.sum(H1))
    sum_H.append(np.sum(H2))
    sum_H.append(np.sum(H3))

    seuil_H1_H3 = 2
    seuil_H0_H1 = 2
    seuil_H0_H2 = 2
    seuil_H2_H3 = 2
    seuil_rot = 80
    seuil_inner = 4
    seuil_zoom = 40
    seuil_rot_var = 0.11

    # print ("RotationZoom seuil: ", compare_angMag)
    # print (sum_H[0]+sum_H[0]+sum_H[0]+sum_H[0])
    # print ("Variance: ", mean_img)
    if (sum_H[1] + sum_H[3]) > seuil_H1_H3 * (sum_H[0] + sum_H[2]):
        if sum_H[1] > seuil_inner * sum_H[3]:
            print("    > Type: Travelling/ tilt vers le coin inférieur gauche (01)")
        elif sum_H[3] > seuil_inner * sum_H[1]:
            print("    > Type: Travelling/ tilt vers le coin supérieur gauche (02)")
        else:
            print("    > Type: Travelling/ tilt vers la gauche (03)")
    elif (sum_H[0] + sum_H[1]) > seuil_H0_H1 * (sum_H[2] + sum_H[3]):
        if sum_H[1] > seuil_inner * sum_H[0]:
            print("    > Type: Travelling/ tilt vers le coin inférieur gauche (04)")
        elif not sum_H[0] <= seuil_inner * sum_H[1]:
            print("    > Type: Travelling/ tilt vers le coin inférieur droite (05)")
        else:
            print("    > Type: Travelling/ tilt vers le bas (06)")
    elif (sum_H[0] + sum_H[2]) > seuil_H0_H2 * (sum_H[1] + sum_H[3]):
        if not seuil_inner * sum_H[2] >= sum_H[0]:
            print("    > Type: Travelling/ tilt vers le coin inférieur droite (07)")
        elif not sum_H[2] <= seuil_inner * sum_H[0]:
            print("    > Type: Travelling/ tilt vers le coin supérieur droite (08)")
        else:
            print("    > Type: Travelling/ tilt vers la droite (09)")
    elif (sum_H[2] + sum_H[3]) > seuil_H2_H3 * (sum_H[0] + sum_H[1]):
        if sum_H[2] > seuil_inner * sum_H[3]:
            print("    > Type: Travelling/ tilt vers le coin supérieur droite (10)")
        elif sum_H[3] > seuil_inner * sum_H[2]:
            print("    > Type: Travelling/ tilt vers le coin supérieur gauche (11)")
        else:
            print("    > Type: Travelling/ tilt vers le haut (12)")
    elif sum_H[0] + sum_H[0] + sum_H[0] + sum_H[0] > seuil_zoom:
        if compare_angMag <= seuil_rot_var:
            print("    > Type: Rotation (13)")
        else:
            print("    > Type: Zoom (14)")
    else:
        print("    > Type: Plan fixe (15)")


# Reads the video from file
# capture = cv2.VideoCapture("../TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
# capture = cv2.VideoCapture("../TP2_Videos/Extrait5-Matrix-Helicopter_Scene(280p).m4v")


# capture = cv2.VideoCapture("../Plan_type/Zoom_1.mp4")
# capture = cv2.VideoCapture("../Plan_type/Zoom_2.mp4")
# capture = cv2.VideoCapture("../Plan_type/Rotation_1.mp4")


# Reads video from webcam
capture = cv2.VideoCapture(0)

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
hog_hist = apply_hog(frame2, number_ori)
plan_selection = []
hog_accumulate = np.zeros_like(apply_hog(frame2, number_ori))

typePlan_moyen = np.zeros_like(Calcule_Flow_histogramme(flow))

##Counter pour le frames
countFrames = 0

# Variables utilisées pour la détéction de plan
plans_count = 0
plans_selection = []
corrPlan_list = []
hist_flow_ori = []
step = 24
flow_hist_descript = np.zeros_like(flow_hist(flow, step))
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
	@ Threshold:	Extrait 1: 0.85
	@ 				Extrait 2:   -
	@ 				Extrait 3:   -
	@ 				Extrait 4:   -
	@ 				Extrait 5: 0.84
	"""
    ## Sauvegard l'ancien et le nouveau histogramme
    yuvHist_old = yuvHist
    yuvHist = Calcule_2D_YUV_histogramme(ret, frame2)
    yuv_toCorr = cv2.compareHist(yuvHist_old, yuvHist, cv2.HISTCMP_CORREL)
    if yuv_toCorr < 0.845:
        detec_yuv = 1

    """
	@ METHODE 2: (Vx,Vy)Histogramme
	@ Threshold:	Extrait 1: 0.2
	@ 				Extrait 2:  -
	@ 				Extrait 3:  -
	@ 				Extrait 4:  -
	@ 				Extrait 5: 0.2
	"""
    ## Calcule le flow de l'image
    flow = Find_flow(prvs, next)
    # https://www.cse.iitb.ac.in/~sharat/icvgip.org/ncvpripg2008/papers/9.pdf

    ## Calcule des histogrammes de angle et magnitude (moyen)
    flow_hist_descript = flow_hist_descript + flow_hist(flow, step)

    ## Compare les histogrammes
    VxVyHist_old = VxVyHist
    VxVyHist = Calcule_Flow_histogramme(flow)
    flow_toCorr = cv2.compareHist(VxVyHist_old, VxVyHist, cv2.HISTCMP_CORREL)
    if flow_toCorr < 0.2:
        detec_flow = 1
    ## Variable utilisé pour la détéction de type de plan
    typePlan_moyen += VxVyHist

    """
	@ METHODE 3: HOG
	@ Threshold:	Extrait 1: 0.95
	@ 				Extrait 2: 0.95
	@ 				Extrait 3: 0.973
	@ 				Extrait 4:   -
	@ 				Extrait 5: 0.9
	"""
    ### Calcule le HOG
    hog_hist_old = hog_hist
    plan_selection.append([frame2, hog_hist])
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    hog_hist = apply_hog(frame2, number_ori)
    hog_hist_toCorr = np.corrcoef(hog_hist_old, hog_hist)[0, 1]
    if hog_hist_toCorr < 0.95:
        detec_hog = 1

    """
	@ DÉTECTION DE PLAN E IMAGE CLEF
	@ 	1. Détéction: fait avec la fonction alpha*detec_yuv + beta*detec_flow + gamma*detec_hog,
	@		ou alpha, beta et gamma sont des coeficients de pondération pour chaque méthode de
	@		détéction
	@	2. Image Clef: Fait en utilisant le HOG pour chaque plan
	"""
    ## Ponderation pour détécter le changement de plan
    alpha = 1
    beta = 1
    gamma = 1
    detec_combine = alpha * detec_yuv + beta * detec_flow + gamma * detec_hog

    ## Accumulateur pour faire obtenir la valeur moyenne de HOG (réprésentatif)
    hog_accumulate += hog_hist

    ## Vote majoritaire pour la détéction de plans
    if detec_combine > 1:
        # if countFrames%27==0:
        if plans_count == 0:
            plans_selection.append([0, countFrames])
        else:
            plans_selection.append([plans_selection[plans_count - 1][1] + 1, countFrames])
        print(plans_count, "- Plan détécté entre les frames: ", plans_selection[plans_count])
        plans_count += 1

        ## Sélection d'une image clef pour le plan détecté
        hog_plan_moyen = hog_accumulate / shape(plan_selection)[0]

        ## Detection du type plan dominant
        typePlan_moyen = typePlan_moyen / shape(plan_selection)[0]

        ## Histogramme de angle et magnitude dominant
        flow_hist_descript = flow_hist_descript / shape(plan_selection)[0]
        flow_hist_descript[0] = flow_hist_descript[0] / np.max(flow_hist_descript[0])
        flow_hist_descript[1] = flow_hist_descript[1] / np.max(flow_hist_descript[1])

        to_sort = np.copy(flow_hist_descript[1][:])
        to_sort.sort()
        to_sort = to_sort[::-1]
        compare_angMag = 0

        ## Vérifier
        for i in range(5):
            mag_sort = flow_hist_descript[0][:]
            angle_sort = flow_hist_descript[1][:]
            compare_angMag += abs(to_sort[i] - float(mag_sort[angle_sort[:] == to_sort[i]]))
        compare_angMag = compare_angMag / 5

        for index in range(shape(plan_selection)[0]):
            corr_plan = np.corrcoef(plan_selection[index][1], hog_plan_moyen)[0, 1]
            corrPlan_list.append(corr_plan)

        ## Image clé sera celle ayant plus de similarité avec le HOG moyen
        index_cle = np.where(corrPlan_list == np.max(corrPlan_list))[0]

        ## Sauvegard l'image clef pour chaque plan
        cv2.imwrite('Image_clef/Plan_%d.png' % plans_count, plan_selection[int(index_cle)][0])

        ## Classification du type de plan
        Classification_type_plan(typePlan_moyen, compare_angMag)

        ### Reinitialise les lists de sélection
        plan_selection.clear()
        corrPlan_list.clear()
        hog_accumulate = 0
        typePlan_moyen = 0

    countFrames += 1

    # Type "q" to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    prvs = next
    ret, frame2 = capture.read()
    if (ret):
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

cv2.destroyAllWindows()
