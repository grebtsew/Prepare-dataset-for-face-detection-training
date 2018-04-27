print ("Import libraries...")

import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from shutil import copyfile

from collections import defaultdict
from io import StringIO
from PIL import Image

import face_recognition
import dlib

import cv2

FaceDetector = None

import create_xml as xml

class file_path:
    def __init__(self):
        self.path = ""
        self.name = ""

print ("Loading dlib Modell...")

FaceDetector = dlib.get_frontal_face_detector()

print ("Get files...")

PATH_TO_TEST_IMAGES_DIR = 'images'
TEST_IMAGE_PATHS = []

valid_images = [".jpg",".gif",".png",".tga"]

for files in os.listdir(PATH_TO_TEST_IMAGES_DIR):
  ext = os.path.splitext(files)[1]
  if ext.lower() not in valid_images:
    continue
  temp = file_path()
  temp.name = files
  temp.path = os.path.join(PATH_TO_TEST_IMAGES_DIR,files)
  TEST_IMAGE_PATHS.append(temp)


# Convert_dlib_box_to_OpenCV_box(box)
# @param takes in a dlib box
# @return returns a box for OpenCV
def convert_dlib_box_to_openCV_box(box):
    return (int(box.left()), int(box.top()), int(box.right() - box.left()),
                         int(box.bottom() - box.top()) )

# Detection and actual loop
print("Detecting...")

# for each image
for image_path in TEST_IMAGE_PATHS:

    frame = cv2.imread(image_path.path)
    height, width, channels = frame.shape


    box_arr = FaceDetector(frame, 0)



    if len(box_arr) > 0 :

        # Convert box to OpenCV
        face_box = convert_dlib_box_to_openCV_box(box_arr[0])

        print(face_box)



      # Log detection result
        print (time.strftime('%d/%m/%Y %H:%M:%S'))


        detect_box_array = []
        ymin = face_box[1]
        xmin = face_box[0]
        ymax = face_box[1]+face_box[2]
        xmax = face_box[0]+face_box[3]

        print ("ymin=%s xmin=%s ymax=%s xmax=%s"  % (ymin, xmin, ymax, xmax ))

        new_box = xml.detected_object()
        new_box.name = 'face'
        new_box.pose = "Unspecified"
        new_box.truncated = 0
        new_box.difficult = 0
        new_box.positions = [ymin, xmin, ymax, xmax]

        detect_box_array.append(new_box);

        # create xml
        xml.create_training_xml('images',
                              image_path.name,
                              os.path.abspath(image_path.path),
                              'Unknown',
                              width,
                              height,
                              len(box_arr),
                              0,
                              detect_box_array,
                              'res')


        topLeft = (int(face_box[0]), int(face_box[1]))
        bottomRight = (int(face_box[0] + face_box[2]), int(face_box[1] + face_box[3]))
        cv2.rectangle(frame, topLeft,bottomRight, (0,0,255), 2,1 )


        print("Created file %s" % (''.join(['res/','/', image_path.name])))
        copyfile(image_path.path, ''.join(['res/',image_path.name]))
        cv2.imshow('face detection', frame)
        cv2.waitKey(0)
    else:
        copyfile(image_path.path, ''.join(['fail/',image_path.name]))
        print("Created file %s" % (''.join(['fail/','/', image_path.name])))
