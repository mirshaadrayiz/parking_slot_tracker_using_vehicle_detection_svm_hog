import numpy as np
import cv2
import HOG
import matplotlib as plt


def extract_features(img, color_space='RGB'):
    img_features = []
    
    if color_space == 'RGB':
        # Convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply square root normalization
        img_gray = np.sqrt(img_gray / float(np.max(img_gray))) * 255
        # Calculate HOG features
        hog_features = HOG.hog(img_gray)
        img_features.append(hog_features)
        print ("Feature Extraction Successful")
    return np.concatenate(img_features)

image = cv2.imread('car.png')
cvtimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
vehicle_features = extract_features(cvtimage, color_space='RGB')
print(vehicle_features)
print(vehicle_features.shape)