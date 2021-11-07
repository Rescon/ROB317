import cv2
import matplotlib.pyplot as plt
import numpy as np

for option in cv2.__dict__:
    if 'CORREL' in option:
        print(option)


def normalize(hist):
    valmax = np.amax(hist)
    valmin = np.amin(hist)
    hist = hist / (valmax - valmin) * 255
    return hist


def plot_corplot_correlationrelation(corr):
    plt.figure(num=1, figsize=(10, 6))
    plt.clf()
    plt.plot(corr, 'b', linewidth=0.5)
    plt.ylim([0, 1])
    plt.title("Correlation des histogram h et h-1")
    plt.xlabel("Numero de frames")
    plt.ylabel("Correlation (%)")
    # plt.savefig('test.jpg')
    plt.draw()
    # plt.pause(0.0001)


# Ouverture du flux video
# cap = cv2.VideoCapture("TP2_Videos/Extrait1-Cosmos_Laundromat1(340p).m4v")
cap = cv2.VideoCapture("../TP2_Videos/Extrait2-ManWithAMovieCamera.m4v")

ret, frame = cap.read()  # Passe à l'image suivante
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
hist = cv2.cvtColor(normalize(hist).astype('float32'), cv2.COLOR_GRAY2BGR)
hist_old = hist
corr = []

cv2.namedWindow('Histogramme', cv2.WINDOW_NORMAL)
# cv2.namedWindow('Video', cv2.WINDOW_NORMAL)
index = 1

while ret:
    index += 1

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = normalize(hist).astype('float32')
    corr.append(cv2.compareHist(hist_old, hist, 0))

    cv2.imshow('Histogramme', hist)
    # cv2.imshow('Video', frame)

    # Plot en temps réel de la valeur de la correlation
    plot_corplot_correlationrelation(corr)

    hist_old = hist
    k = cv2.waitKey(15) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('Frame_%04d.png' % index, frame)
        cv2.imwrite('YUV_Frame_%04d.png' % index, gray)
    elif k == ord('q'):
        break
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

cap.release()
cv2.destroyAllWindows()
