print ("Import libraries...")

import numpy as np
import os, os.path
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
from shutil import copyfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import face_recognition
import dlib

import cv2

FaceDetector = None



# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util

from utils import visualization_utils as vis_util

from Own_Code import create_xml as xml


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

class file_path:
    def __init__(self):
        self.path = ""
        self.name = ""
       

# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
#DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
#PATH_TO_CKPT = "frozen_inference_graph_face.pb"#MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

#NUM_CLASSES = 90

# ## Load a (frozen) Tensorflow model into memory.

print ("Loading Modell...")

FaceDetector = dlib.get_frontal_face_detector()
 
#detection_graph = tf.Graph()
#with detection_graph.as_default():
#  od_graph_def = tf.GraphDef()
#  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#    serialized_graph = fid.read()
#    od_graph_def.ParseFromString(serialized_graph)
#    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

#print ("Loading label map...")


#label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
#category_index = label_map_util.create_category_index(categories)


print ("Get files...")

# For the sake of simplicity we will use only 2 images:
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'inviduell_del/bilder(50)'
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

# # Detection and actual loop

print("Detecting...")

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
        new_box.name = 'face'#category_index[np.squeeze(classes).astype(np.int32)[i]]['name']
        new_box.pose = "Unspecified"
        new_box.truncated = 0
        new_box.difficult = 0
        new_box.positions = [ymin, xmin, ymax, xmax]

        detect_box_array.append(new_box);

        xml.create_training_xml('inviduell_del/bilder(50)',
                              image_path.name,
                              os.path.abspath(image_path.path),
                              'Unknown',
                              width,
                              height,
                              len(box_arr),
                              0,
                              detect_box_array,
                              'inviduell_del/dataset(50)')

              
        topLeft = (int(face_box[0]), int(face_box[1]))
        bottomRight = (int(face_box[0] + face_box[2]), int(face_box[1] + face_box[3]))
        cv2.rectangle(frame, topLeft,bottomRight, (0,0,255), 2,1 )


        print("Created file %s" % (''.join(['inviduell_del/dataset(50)','/', image_path.name])))
        copyfile(image_path.path, ''.join(['inviduell_del/dataset(50)/',image_path.name]))
        cv2.imshow('object detection', frame)
        cv2.waitKey(1)
    else:
        copyfile(image_path.path, ''.join(['inviduell_del/nodetection(50)/',image_path.name]))
        print("Created file %s" % (''.join(['inviduell_del/nodetection(50)','/', image_path.name]))) 
