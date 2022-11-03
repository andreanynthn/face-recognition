import os
import cv2
import uuid
import numpy as np
from PIL import Image
import streamlit as st
from faceTraining import faceTraining


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
le = faceTraining('dataset')[2]


def loadDatabase(filename):
    """
    :param filename: name of database file
    :return: list of name
    """

    database = open(filename, 'r')

    data = database.read().split('\n')
    # names = [name for name in [data[i].split("'")[1] for i in range(len(data)-1)]]
    names = [name for name in [data[i].split("'")[1] for i in range(len(data)-1)]]
    names.insert(0, 'None')

    return names

def main():
    st.title("Face Recognition")

    image_file = st.file_uploader(
        "Upload image", type = ["jpg", "png", "jpeg"]
    )

    if image_file is not None:
        image = Image.open(image_file)

        if st.button("Process"):

            # process image

            img = np.array(image.convert('RGB'))
            img = cv2.cvtColor(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray,
                                                   scaleFactor = 1.2,
                                                   minNeighbors = 5,
                                                   minSize = (64, 48)
                                                   )
            path = 'dataset'
            executionPath = os.getcwd()
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer.yml')
            detector = "haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(detector)
            names = loadDatabase('database.txt')

            for(x,y,w,h) in faces:
                face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                # print(face_id)

                if (round(100 - (confidence)) >= 50):
                    # name = names[face_id]
                    name = le.inverse_transform([face_id])[0]
                    confidence_score = "  {0}%".format(round(100 - confidence))
                    st.success("Face found!")
                    st.image(img, caption = f"Name: {name}; Confidence score: {confidence_score}")
                else:
                    st.error("Face not found!")

if __name__ == '__main__':
    main()