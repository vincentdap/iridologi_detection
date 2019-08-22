# Iridologi Detection

Iridology is a scientific study of the shape and structure of iris can provide an overview of every organ in human body. Research on computerized iridology has been carried out. Cases of decreased organ excretion through iridology that are commonly found are lung and kidney organs. The purpose of this study use Convolutional Neural Network to detect decreased organ function in lungs and kidneys through the iris of the eye. The study of iridology and iris image obtained from the iridologist Dr. Asdi Yudiono at Intan Clinic, Pakualaman, Yogyakarta. The cropping method is used to extract the identified part of the eye image. The cropping method consists of a median filter to remove noise, a hough circle transform to get an iris circle and a region of interest to get the identified part. Image cropping results are used as training data and test data. The Convolutional Neural Network training process uses the VGG16 model with 2 classes, normal and not normal. The results of Convolutional Neural Network research can detect decreased organ function in excretion through the iris of the eye. From 40 testing data with details of 20 right eyes and 20 left eyes, the accuracy is 90%.

To run the experiments, you would need:

* CUDA device with global RAM >= 4 GB (tested with GTX 1050 4GB)
* Python 2.7.x
* Virtualenv (Anaconda) (optional. For isolated environment)
* Tensorflow Object Detection

> The programs can also run on CPU. You just need to use `tensorflow` instead of `tensorflow-gpu`. Further info 
1. https://www.tensorflow.org/install/
2. https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

### How to run :
python main.py
