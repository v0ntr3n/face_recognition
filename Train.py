import os

import cv2
import numpy as np

DATA_PATH = "images"


dict = {}
labels = []
for i in os.listdir(DATA_PATH):
    dict[i] = len(dict)
    labels.append(i)

# Save labels
with open("labels.txt", 'w+') as file:
    file.write('|'.join(labels))




X_train = []
y_train = []

for users in os.listdir(DATA_PATH):
    users_path = os.path.join(DATA_PATH, users)
    lst_user = []

    for image in os.listdir(users_path):
        image_path = os.path.join(users_path, image)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        lst_user.append(img)
        y_train.append(dict[users])

    X_train.extend(lst_user)


recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Training...")

recognizer.train(X_train, np.array(y_train))
recognizer.write("data/recognition.yml")
print("Done")
