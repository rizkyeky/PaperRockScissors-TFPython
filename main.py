import cv2
# import numpy as np

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    x = 15
    y = 15
    h, w, c = frame.shape

    frame = frame[x:h-x, y:w] 
    frame = cv2.resize(frame, (int(frame.shape[1]*0.3), int(frame.shape[0]*0.3)))

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

def process_img(img):
  # img = cv2.imread("/content/0Flw60Z2MAWWKn6S.png")
  # cv2.imshow(img)

  x = 15
  y = 15
  h, w, _ = img.shape

  img = img[x:h-x, y:w]

  _, img = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY)
  # cv2.imshow(img)

  maskGreen = cv2.inRange(img, (0, 255, 0), (0, 255, 0))
  img[maskGreen != 0] = [0,0,0]

  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # cv2.imshow(img)

  _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
  # cv2.imshow(img)

  img = cv2.resize(img, (int(img.shape[1]*0.3), int(img.shape[0]*0.3)))
  cv2.imshow(img)
  # print(img.shape)
  return img
  