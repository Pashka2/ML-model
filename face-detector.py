import cv2

#load some pretrained data on face frontels from opencv ( haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('harrcascade_frontalface_default.xml')

#filename = "C:\Users\pasha\OneDrive\Desktop\PYTHON PROGRAMS\ML model\rdj.jpg"

#choose an image to detect faces in
img = cv2.imread("rdj face.jpg")



cv2.imshow("Pasha ",img)

cv2.waitKey()
print("code completed")