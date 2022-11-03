''''
Training Multiple Faces stored on a DataBase:
	==> Each face should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model will be saved on trainer/ directory. (if it does not exist, pls create one)
	==> for using PIL, install pillow library with "pip install pillow"

Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18   

'''
import re
import cv2
import numpy as np
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder

# Path for face image database
# path = 'dataset'

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

# function to get the images and label data
def faceTraining(path):

    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    le = LabelEncoder()

    imagePaths = [os.path.join(path,f) for f in sorted_alphanumeric(os.listdir(path))]
    faceSamples=[]
    labels = []
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
        img_numpy = np.array(PIL_img,'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[0])
        label = os.path.split(imagePath)[-1].split(".")[1]
        faces = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h,x:x+w])
            labels.append(label)
            ids.append(id)

    le.fit(labels)
    labels_enc = le.transform(labels)

    # recognizer.train(faceSamples, np.array(ids))
    return faceSamples, labels_enc, le
    # return recognizer.write('trainer.yml')
    # return recognizer

recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, labels_enc, le = faceTraining('dataset')
recognizer.train(faces, labels_enc)
recognizer.write('trainer.yml')
# print(imagePaths)
# print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
# faces,ids = getImagesAndLabels(path)
# recognizer.train(faces, np.array(ids))
#
# # Save the model into trainer/trainer.yml
# recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi
#
# # Print the numer of faces trained and end program
# print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
