import cv2

from utils.pre_process import pre_processing

FPS = 30

recognize = cv2.face.LBPHFaceRecognizer_create()
recognize.read("data/recognition.yml")

cascadepath = "data/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadepath)

font = cv2.FONT_HERSHEY_SIMPLEX
labels = open("labels.txt").read().split("|")

# http protocol
# cam = cv2.VideoCapture("http://admin:nice@192.168.0.100:8081")

# rtsp protocol
cam = cv2.VideoCapture("rtsp://admin:nice@192.168.0.100:8554/live")
cam.set(cv2.CAP_PROP_FPS, FPS)


while True:
    ret, img = cam.read()
    gray = pre_processing(img)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognize.predict(gray[y : y + h, x : x + w])
        confidence = " {0}%".format(round(100 - confidence))

        if confidence < 100:
            id = labels[id]
        else:
            id = "unknown"

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 80, y - 5), font, 1, (255, 255, 0), 2)

    cv2.imshow("RECOGNIZE", img)

    if cv2.waitKey(1) == 27:
        break

print("\nEXIT")
cam.release()
cv2.destroyAllWindows()
