# vehicle_detection_using_svm

## OBJECTIVES

1. Preprocess images using multiple techniques to make them ready for feature extraction
2. Extract features from images of vehicles and non-vehicles using feature extraction techniques
3. Train a linear SVM classifier using extracted features
4. Test the classifier on new unseen data
5. Evaluate the model for accuracy and other scores


**NOTE** : A custom HOG algorithm implementation will be used for HOG extraction. Link given below
          [Histogram of Gradients](https://github.com/mirshaadrayiz/histogram-of-gradients)
 
 ## INSTALLATIONS REQUIRED

  1. Python : https://www.python.org/downloads/
  2. OpenCV : https://pypi.org/project/opencv-python/
  3. Numpy : https://docs.scipy.org/doc/numpy-1.10.1/user/install.html
  4. matplotlib : https://matplotlib.org/faq/installing_faq.html
  5. scikit-learn : https://scikit-learn.org/stable/
  6. scikit-image : https://scikit-image.org/


## DATASET

[VEHICLE](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [NON-VEHICLE](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) data set provided by Udacity is used.

No. of positive images = 5966  
No. of negative images = 5354  
Image shape = (64,64,3)

###### Samples of positive images
![image](https://github.com/mirshaadrayiz/vehicle_detection_using_svm/assets/147004775/5ab100f9-3121-4e22-8d62-4c4a6ca23ed7)

###### Samples of negative images
![image](https://github.com/mirshaadrayiz/vehicle_detection_using_svm/assets/147004775/8c2d0fe7-fec3-455d-89f0-0c8c6414f97d)


## DATA PREPARAION

For diversity in dataset, data augmentation was performed. Data augmentation is however optional here as we have a vast amount of data for a simple classifier. 

`np.fliplr()` - Flip image  
`cv2.warpAffine()` - Affine transformation to shift images left, right, up and down

## FEATURE EXTRACTION

`spatial_features()`  - scales down each color channel of the image and flattens them into a single 1D feature vector capturing spatial information from the image.  

`color_histogram(img)` - computes a color histogram  

`HOG.hog()` - computes the histogram of gradients using custom HOG algorithm (Refer [Histogram of Gradients](https://github.com/mirshaadrayiz/histogram-of-gradients) for more info)

While computing HOG, better results could be obtained by changing parameters like cell size, block size, angle unit, pixels per cell etc., but this could increase computational complexity. I decided that the tradeoff in complexity for a smaller increase in performance was not worth it for this application.

###Optional  

`harriscorner()` - computes the corners features of objects in an image.  
Harris Corner was implemented, but not used in this project to avoid overfitting. Increased features increase complexity of the model resulting in overfitting.

##### Visualization of HOG applied on an image

![image](https://github.com/mirshaadrayiz/vehicle_detection_using_svm/assets/147004775/c4d5798f-0ae7-4e6f-9520-97af473bc1a1)

`extract_features()` - Performs all above mentioned feature extraction techniques for all input images and returns a feature vector


## TRAINING THE CLASSIFIER

The classifier used in this project was **Support Vector Machines (SVM)**. SVMs are known to perform well for classification problems.

`StandardScaler()` - Perform feature vector normalization on the obtained feature vectors
`train_test_split(scaled_X, y, test_size=0.05) ` - Splitting the scaled feature vector into train and test sets.

The model was trained using the training data and was validated for accuracy.


## VALIDATING THE CLASSIFIER

ROC curves were generated to identify the impact of each feature extraction technique

#### Only HOG

![image](https://github.com/mirshaadrayiz/vehicle_detection_using_svm/assets/147004775/5378be72-ff4e-4d12-a7e1-5e97c6c3a348)

#### HOG with color features

![image](https://github.com/mirshaadrayiz/vehicle_detection_using_svm/assets/147004775/b27a823a-f6cd-47c6-b61f-3782209ded55)

#### HOG with color and spatial features 

![image](https://github.com/mirshaadrayiz/vehicle_detection_using_svm/assets/147004775/0aa747b4-3640-4dca-9e94-0f3e42fbc94e)

A model with accuracy **0.96** was obtained.

