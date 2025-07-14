import os

import cv2

from utils.pre_process import pre_processing

PATH = "images"
face_detector = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

StudentID = input("\n\nInput your StudentID:")

target_path = os.path.join(PATH, StudentID)
if not os.path.isdir(target_path):
    os.mkdir(target_path)


print("\nCreate Camera...")

# rtsp protocol
cap = cv2.VideoCapture("rtsp://admin:nice@192.168.0.100:8554/live")

# http protocol
# cap = cv2.VideoCapture("http://admin:nice@192.168.0.100:8081")

count = 1


while True:
    _, img = cap.read()
    img_pre = pre_processing(img)
    faces = face_detector.detectMultiScale(img_pre, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        img_face = cv2.resize(
            img_pre[y : y + h, x: x + w], [100, 100]
        )

        image_path = os.path.join(target_path, "anh_{}.jpg".format(count))
        print(image_path)
        cv2.imwrite(image_path, img_face)
        count += 1

    cv2.imshow("frame", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # esc
        break

print("\nExit")
cap.release()
cv2.destroyAllWindows()
