from module import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", '--input', required=True, help="Diretório da imagem/Arquivo TXT contendo as imagens")
args = vars(ap.parse_args())

filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

if "text/plain" == filetype:
    imagePaths = open(args["input"]).read().strip().split("\n")

print("[INFO] Carregando Modelo Detector de Lixo...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())

CONFIDENCE_THRESHOLD = 0

for imagePath in imagePaths:
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    (boxPreds, labelPreds) = model.predict(image)
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)

    for i in range(0, boxPreds.shape[1], 4):
        (startX, startY, endX, endY) = boxPreds[0, i:i+4]
        confidence = np.max(labelPreds)

        if confidence >= CONFIDENCE_THRESHOLD:
            j = np.argmax(labelPreds, axis=1)[0]
            label = lb.classes_[j]
            
            (h, w) = image.shape[:2]

            startX = int(startX * w)
            startY = int(startY * h)
            endX = int(endX * w)
            endY = int(endY * h)

            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(0)