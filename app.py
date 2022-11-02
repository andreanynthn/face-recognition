import os
import cv2
import uuid
import numpy as np
from PIL import Image
import streamlit as st


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# create 'dataset' directory
# if 'dataset' in os.listdir(os.getcwd()):
#     pass
# else:
#     os.mkdir(os.path.join(os.getcwd(), 'dataset'))


def loadDatabase(filename):
    """
    :param filename: name of database file
    :return: list of name
    """

    database = open(filename, 'r')

    data = database.read().split('\n')
    # names = [name for name in [data[i].split("'")[1] for i in range(len(data)-1)]]
    names = [name for name in [data[i].split("'")[3] for i in range(len(data)-1)]]
    names.insert(0, 'None')

    return names


def faceTraining(path):

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)

    recognizer.train(faceSamples, np.array(ids))

    # return recognizer.write('trainer/trainer.yml')
    return recognizer

def main():
    st.title("Face Recognition")

    # menu = ['Face Registration', 'Face Recognition']
    # choice = st.sidebar.selectbox("Menu", menu)


    # face registration
    # if choice == "Face Registration":
    #     st.subheader("Face Registration")
    #
    #     image_file = st.file_uploader(
    #         "Upload image", type = ["jpg", "png", "jpeg"]
    #     )
    #
    #     if image_file is not None:
    #
    #         username = st.text_input("Input name")
    #         faceId = st.text_input("Input id")
    #         image = Image.open(image_file)
    #
    #         if st.button("Process"):
    #
    #             # process image
    #
    #             img = np.array(image.convert('RGB'))
    #             img = cv2.cvtColor(img, 1)
    #             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #             faces = face_detector.detectMultiScale(gray, 1.3, 5)
    #
    #             ids = open('faceId.txt', 'r')
    #             listId = ids.read().split('\n')
    #
    #             for (x,y,w,h) in faces:
    #                 cv2.rectangle(img, (x,y), (x+w+2,y+h+2), (255,0,0), 2)
    #
    #             if faceId in listId:
    #                 st.error("Id already registered.")
    #             else:
    #                 with open('database.txt', 'a') as f:
    #                     f.write(f'{faceId, username}\n')
    #
    #                 with open('faceId.txt', 'a') as f:
    #                     f.write(f'{faceId}\n')
    #                 cv2.imwrite(f"dataset/{username}." + faceId + ".jpg", gray[y:y+h,x:x+w])
    #                 st.success("Successfully registered face.")


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
            # executionPath = os.getcwd()
            recognizer = faceTraining(path)
            recognizer.read('trainer.yml')
            detector = "haarcascade_frontalface_default.xml"
            faces = cv2.CascadeClassifier(detector)
            names = loadDatabase('database.txt')

            for(x,y,w,h) in faces:
                face_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                if (round(100 - (confidence)) >= 50):
                    name = names[face_id]
                    confidence_score = "  {0}%".format(round(100 - confidence))
                    st.success("Face found!")
                    st.image(img, caption = f"Name: {name}; Confidence score: {confidence_score}")
                else:
                    st.error("Face not found!")

if __name__ == '__main__':
    main()
