import cv2

img2 = cv2.imread("rockpaperscissors/paper/ZveWRNmdKejc1c4w.png", flags=cv2.IMREAD_GRAYSCALE)

cv2.imshow('img', img2)

cv2.waitKey(0)


