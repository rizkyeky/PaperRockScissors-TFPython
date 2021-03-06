import cv2

cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

cropX = 50
cropY = 50

while(True):

    ret, frame = cap.read()
    height, width, c = frame.shape

    frame = cv2.resize(frame, (int(width*0.5), int(height*0.5)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)

    faces = faceCascade.detectMultiScale(frame)
    for (x, y, w, h) in faces:
        color = (255, 0, 0)
        stroke = 2

        xCord = x + w
        yCord = y + h
        cv2.rectangle(frame, (x, y), (xCord, yCord), color, stroke)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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
    img[maskGreen != 0] = [0, 0, 0]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(img)

    _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    # cv2.imshow(img)

    img = cv2.resize(img, (int(img.shape[1]*0.3), int(img.shape[0]*0.3)))
    cv2.imshow(img)
    # print(img.shape)
    return img
