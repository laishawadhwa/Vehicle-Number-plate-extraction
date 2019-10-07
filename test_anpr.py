import numpy as np
import os
import sys
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import pytesseract
import matplotlib.patches as patches
from utilities import allow_needed_values as anv 
from utilities import do_image_conversion as dic
import cv2

MODEL_NAME = 'numplate'
PATH_TO_CKPT = MODEL_NAME + '/graph-200000/frozen_inference_graph.pb'
PATH_TO_LABELS = 'object-detection.pbtxt'
NUM_CLASSES = 1


detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')



def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'png_tesseract/test_motion'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 14) ]
IMAGE_SIZE = (12, 8)
TEST_DHARUN=os.path.join('numplate')
count = 0


with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    it = 0
    #video = cv2.VideoWriter('png_tesseract/data/output2.avi', 0, 1, (1280, 720))
    out = cv2.VideoWriter('png_tesseract/data/output.avi',cv2.VideoWriter_fourcc(*'DIVX'), 8, (1280, 720))
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path) 
      print(image_path)
      image_np = load_image_into_numpy_array(image)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      #print("Number of detections : ", num)
      ymin = boxes[0,0,0]
      xmin = boxes[0,0,1]
      ymax = boxes[0,0,2]
      xmax = boxes[0,0,3]
      (im_width, im_height) = image.size
      (xminn, xmaxx, yminn, ymaxx) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
      print("Plate detected: ", (xminn, xmaxx, yminn, ymaxx))
      
     
      
      
        
      out.write(image_np)
    
     
      plt.imshow(image_np)
      plt.show()
      it = it+1
      

      cropped_image = tf.image.crop_to_bounding_box(image_np, int(yminn), int(xminn),int(ymaxx - yminn), int(xmaxx - xminn))
      img_data = sess.run(cropped_image)
      count = 0
      filename = dic.yo_make_the_conversion(img_data, count)
      pytesseract.tesseract_cmd = 'Users/laishawadhwa/models/research/object_detection/tessdata/'
      text = pytesseract.image_to_string(Image.open(filename),lang=None) 
      print('CHARCTER RECOGNITION : ',anv.catch_rectify_plate_characters(text))

      cv2.rectangle(image_np, (xminn, yminn), (xmaxx, ymaxx), (0, 255, 0) , 2)
      y = xminn - 15 if xminn - 15 > 15 else yminn + 15
      cv2.putText(image_np,text, (xminn - 145 , yminn -50),cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
      print("Text Detected:", text)
      plt.imshow(img_data)
    out.release()
     
