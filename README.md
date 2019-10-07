# Vehicle-Number-plate-extraction
An end-end system to extract number plate characters for indian vehicles.

The project has been developed using TensorFlow for License Plate detection from a given car image and uses the Tesseract Engine to recognize the charactes from the detected plate.

### Software Packs Needed

* <a href='https://www.anaconda.com/download/'>Anaconda 3</a> (**Tool comes with most of the required python packages along with python3 & spyder IDE**)<br>
* <a href='https://github.com/tesseract-ocr/tesseract'>Tesseract Engine</a> (**Must need to be installed**)<br>

### Python Packages Needed

* <a href='https://github.com/tensorflow/tensorflow'>Tensorflow</a><br>
* <a href='https://github.com/skvark/opencv-python'>openCV</a><br>
* <a href='https://github.com/madmaze/pytesseract'>pytesseract</a><br>
* <a href='https://github.com/tzutalin/labelImg'>labelImg</a><br>
* <a href='https://github.com/mdbloice/Augmentor'>Augmentor</a><br>

### Data Prepration

* The given data had 237 images which were less to trian a robust model.
* Augmentor library was used to augment the dataset using the following augmentations:
<ul>
<li>Horizontal flip</li>
<li>Rotations ( 30 – 180)</li>
<li>Random distortions</li>
<li>Skew of two different types</li>
<li>Size preserving shearing</li>
</ul>
* The dataset for training had 550 images after augmentation

#### TRAINING PHASE -- IMAGE LABELING

* Generated the set of 500 images (Cars along with number plate). Then annotated the set of images by drawing the boundary box over the number plates to send it for the training phase as Augmentor library had.
  * The Annoation gives the co-ordinates of license plates such as **(xmin, ymin, xmax, ymax)**
  * Then the co-ordinates are saved into a **XML** file by Augmentor library.
  * All the XML files are grouped and the Co-ordinates are saved in **CSV** file.
  * Then the CSV file is converted into **TensorFlow record format**.
* The set of other separate 10 images also gone through the above steps and saved as **Test Record file** 

#### GPU TRAINING

* **Tensorflow-gpu** versionwas used to send the set of annotated images were sent into the **YOLOv3** network (Tf object detection API was used), where the metrics such as model learning rate, batch of images sent into the network and evaluation configurations were set. The training For object detection was done on Colab.
For OCR: tesseract trained on a dataset of 35000 low quality digits dataset.

#### OCR PART

* Then the detected number plate is cropped using Tensorflow, By using the Google **Tesseract-OCR** (Package originally developed to scan hard copy documents to filter out the characters from it) the picture undergoes some coversions using **computer vision** package then the charcters are filtered out.


#### MOTION DETECTION PART

* The basic motion capturing has been implemented to capture the picture of moving vehicle by using the **openCV** library where the sampling rate (i.e. fps) is taken as input for running the application and inference is performed on each frame



