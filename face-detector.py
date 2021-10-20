import cv2
import numpy as np

#load some pretrained data on face frontels from opencv ( haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("C:\\Users\\pasha\\AppData\\Local\\Programs\\Python\\Python39\\Lib\\site-packages\\cv2\\data\\harrcascade_frontalface_default.xml")

#choose an image to detect faces in
img = cv2.imread('C:/Users/pasha/OneDrive/Desktop/PYTHON PROGRAMS/ML model/download.jpg')

#must convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_corrdinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_corrdinates)

#
#cv2.imshow("Face detector ",grayscaled_img)
cv2.waitKey()


print("code completed")