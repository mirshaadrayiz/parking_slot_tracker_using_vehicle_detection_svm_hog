# vehicle_detection_using_svm

## Objectives

1. Preprocess images using multiple techniques to make them ready for feature extraction
2. Extract features from images of vehicles and non-vehicles using feature extraction techniques
3. Train a linear SVM classifier using extracted features
4. Test the classifier on new unseen data
5. Evaluate the model for accuracy and other scores


### DATASET

[VEHICLE](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [NON-VEHICLE](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) data set provided by Udacity is used.

No. of positive images = 5966
No. of negative images = 5354
Image shape = (64,64,3)

Below custom HOG algorithm iplementation will be used for HOG extraction

(https://github.com/mirshaadrayiz/histogram-of-gradients)https://github.com/mirshaadrayiz/histogram-of-gradients
