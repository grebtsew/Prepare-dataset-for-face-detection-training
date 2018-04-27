# Prepare dataset for face detection training

This python code prepare images from a dataset to be trained in tensorflow.

We use dlib to detect faces in images from all files in folder "images". Then generate a xml document for each file with information about the detections. Finally the result is saved in another folder "res". If no detections found, save copy to folder "fail".


# Requirements:

dlib
cv2 (to show images, can remove cv2 code if unwanted)
