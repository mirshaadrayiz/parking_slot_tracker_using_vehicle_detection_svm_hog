"""
This script is used to calibrate the camera based on the provided images
The distortion coefficients and camera matrix are saved for later reuse
"""
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from settings import CALIB_FILE_NAME

def calibrate(filename, silent = True):
    images_path = 'camera_cal'
    x = 9
    y = 6

    # Object Points (x,y,z coordinates of the chess board), Assuming object points will be same for all images
    objp = np.zeros((y*x, 3), np.float32) #Create an array of zeros

    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2) #Meshgrid
    print(objp[:, :2])
    image_points = []
    object_points = []

    # Defining Criteria
    # Whenever 30 iterations of the algorithm is ran, or an accuracy of epsilon = 0.001 is reached, stop the algorithm and return the answer
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Loop through provided Images
    for image_file in os.listdir(images_path):
        if image_file.endswith("jpg"):
            # turn images to grayscale and find chessboard corners
            img = mpimg.imread(os.path.join(images_path, image_file))
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #Color Conversion
            found, corners = cv2.findChessboardCorners(img_gray, (x, y)) #Find Chess Board Corners
            if found:
                # Make fine adjustments to the corners so higher precision can be obtained before
                corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
                image_points.append(corners2)
                object_points.append(objp)
                '''
                img = cv2.drawChessboardCorners(img, (x, y), corners, found)
                cv2.imshow('img', img)
                cv2.waitKey()
                '''

    # pefrorm the calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_gray.shape[::-1], None, None)
    plt.show()
    img_size  = img.shape
    # pickle the data and save it
    calib_data = {'cam_matrix':mtx,
                  'dist_coeffs':dist,
                  'img_size':img_size}
    with open(filename, 'wb') as f:
        pickle.dump(calib_data, f)


    for image_file in os.listdir(images_path):
        if image_file.endswith("jpg"):
            # show distorted images
            img = mpimg.imread(os.path.join(images_path, image_file))

            undistorted_image = cv2.undistort(img, mtx, dist)
            f = plt.figure()
            f.add_subplot(1,2,1)
            plt.imshow(img)
            f.add_subplot(1, 2, 2)
            plt.imshow(undistorted_image)
            plt.show()

    return mtx, dist

if __name__ == '__main__':
    calibrate(CALIB_FILE_NAME, True)








