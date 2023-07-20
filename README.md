# DriverDrowsiness_DL

Introduction on Drowsiness Detection System

Drowsiness detection systems have become increasingly important in preventing road accidents caused by driver fatigue. With long drives, it is common for drivers to become drowsy and even fall asleep behind the wheel. In this article, we will explore the development of a drowsiness detection system that can promptly alert the driver when signs of drowsiness are detected.

Drowsiness is identified by using vision-based techniques like eyes detection, yawning, and nodding. When it comes to yawning and nodding some people can sleep without yawning and nodding.

# Requirements
●OpenCV: OpenCV is a great tool for image processing and performing many computer vision tasks. It is an open-source library that can be used to perform tasks like face detection, object tracking, and many more tasks.

●TensorFlow: Tensorflow is a free and open-source library, developed by the Google Brain team for machine learning and artificial intelligence. Tensorflow has a particular focus on the training and inference of deep neural networks.

●Keras: Keras is an open-source software library and it provides a Python interface for artificial neural networks. Keras is more user-friendly because it is an inbuilt python library.

# Datasets

The first dataset used is the YawDD Video dataset, which contains videos recorded by a camera mounted on a car dashboard. The dataset consists of male and female drivers with some wearing glasses and others without. 
The second dataset used is the Closed Eyes in the Wild (CEW) dataset. This dataset contains 2423 subjects, with 1192 people having closed eyes and 1231 people with open eyes. Images of open eyes were taken from the Labeled Face in the Wild dataset.

# CODE File
Final_Project_Driver_Drowsiness_Detection.ipynb

# Implementation
The provided code demonstrates the implementation of a driver drowsiness detection system. It utilizes image processing techniques and machine learning to detect signs of drowsiness in drivers. The code imports necessary libraries such as OpenCV, TensorFlow, and Keras for image processing, machine learning, and neural network implementation. The code also downloads datasets from Kaggle related to driver drowsiness detection.

The code includes functions for accessing and processing the image dataset. It loads the images, detects faces using Haar cascades, crops and resizes the detected regions of interest, and assigns class labels to the images. The processed images and their corresponding labels are stored in a list for further analysis.

This code performs driver drowsiness detection using a deep learning model. Here's a summary of the code:
1. The code starts by defining two functions: face_for_yawn() and get_data(). face_for_yawn() detects faces in the "yawn" and "no_yawn" image categories using OpenCV, resizes the detected regions of interest to 145x145 pixels, and stores the resized image arrays along with their class labels in a list called yaw_no. get_data() reads in driver drowsiness detection images from a specified directory, resizes each image array to 145x145 pixels, and appends it with its corresponding class label to a list called data. The function returns the data list containing all the resized images with their corresponding class labels for all the images in the specified directory.
2. The append_data() function combines the yaw_no list (obtained from face_for_yawn()) with the data list (obtained from get_data()) using the extend() method. This creates a single dataset that contains both yawning and non-yawning face images, along with open and closed eye images.
3. The code then creates feature and label lists from the combined dataset. The features are extracted from the dataset and stored in the X list, while the labels are stored in the y list.
4. The feature array X is reshaped to be compatible with the input shape of the deep learning model. It is reshaped to have dimensions (-1, 145, 145, 3), where -1 represents the number of samples, 145x145 represents the image size, and 3 represents the number of color channels (RGB).
5. The labels y are one-hot encoded using scikit-learn's LabelBinarizer to convert them into a binary matrix representation.
6. The data is split into training and testing sets using the train_test_split() function from scikit-learn. The training set comprises 70% of the data, while the testing set comprises 30%.
7. The code then defines a deep learning model using the Keras Sequential API. The model consists of multiple convolutional layers with ReLU activation and max-pooling layers. The output from the convolutional layers is flattened, followed by a dropout layer to prevent overfitting. The model ends with several dense layers with ReLU activation, and a softmax output layer with 4 units corresponding to the 4 classes. The model is compiled with categorical cross-entropy loss, accuracy metric, and the Adam optimizer.
8. Data augmentation is performed using the ImageDataGenerator class from Keras. Augmentation techniques such as rescaling, zooming, horizontal flipping, and rotation are applied to the training data.
9. The model is trained using the fit() function. The training data is passed through the train_generator, and the testing data is passed through the test_generator. The model is trained for 50 epochs.

In summary, this code combines yawning and non-yawning face images with open and closed eye images to create a dataset for driver drowsiness detection. It then trains a deep learning model on this dataset using data augmentation techniques.

# Results
![image](https://github.com/Stan-Batman/DriverDrowsiness_DL/assets/31034647/3f7b1d74-6b57-4d85-bb1a-afe28dd9645d)

The proposed driver drowsiness detection system was trained and evaluated on two datasets: YawDD and CEW.
The maximum testing accuracy achieved on the YawDD dataset was 97.06%, and the maximum training accuracy achieved was 97.62%. The minimum training loss was
0.0628, and the minimum testing loss was 0.0755. The trained model was also evaluated on test images and the accuracy was found to be 96.02% and the loss was 0.1072.

# Conclusion
1. Accuracy ( Training vs Testing) Plot

![image](https://github.com/Stan-Batman/DriverDrowsiness_DL/assets/31034647/0646f298-be28-48a5-8120-39980e461b5d)

2. Loss ( Training vs Testing) Plot

![image](https://github.com/Stan-Batman/DriverDrowsiness_DL/assets/31034647/52c6c9ec-f81a-491f-974e-d2e65dc4db7a)



# List of Dependencies

1. requests
2. IPython
3. numpy
4. pandas
5. os
6. cv2 (OpenCV)
7. opendatasets
8. matplotlib
9. sklearn
10. tensorflow
11. keras
12. tabulate
13. playsound
14. google.colab
15. base64
16. visualkeras

Please make sure to install these dependencies before running the code to ensure that all the required packages are available.
