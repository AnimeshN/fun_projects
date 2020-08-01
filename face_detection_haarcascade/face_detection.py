import cv2

trained_face_data = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')
# img = cv2.imread('zoom_image.jpg')
# img = cv2.imread('ceo.jpeg')
# img = cv2.resize(img, (0, 0), fx=.5, fy=.5)
webcam = cv2.VideoCapture(0)

while True:
    success, frame = webcam.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grey_frame)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, x+h), (0, 255, 0), 2)

    cv2.imshow('Animesh', frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
