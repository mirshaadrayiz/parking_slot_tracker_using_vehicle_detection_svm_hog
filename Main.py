import numpy as np
from numpy import savetxt
import cv2
from skimage.feature import hog
import glob
import time
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import math
import sklearn.metrics as metrics
import HOG
import ColorHist
import spatial

'''
def harriscorner(image):

    gray_img = np.copy(image);
    gray_img = np.float32(gray_img)
    gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY);
    dst = cv2.cornerHarris(gray, 2, 5, 0.09)

    # Results are marked through the dilated corners
    dst = cv2.dilate(dst, None)

    # Reverting back to the original image,
    # with optimal threshold value
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    dst_norm_scaled = cv2.convertScaleAbs(dst_norm)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if int(dst_norm[i,j]) > 100:
                cv2.circle(image, (j,i), 1, (255,0,0), 1)


    # the window showing output image with corners
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(gray_img.astype('uint8'))
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image.astype('uint8'))
    plt.show()
    # De-allocate any associated memory usage
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

    return dst;
    '''


def extract_features(imgs, color_space='RGB', single_image=False):
    if single_image == True:
        img_features = []
        image = np.copy(imgs)

        spatial_features = spatial.spatial(image, (32, 32))
        img_features.append(spatial_features)

        imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2LUV)
        imgs = np.sqrt(imgs)
        hist_features = ColorHist.color_histogram(imgs)
        img_features.append(hist_features)

        image2 = np.copy(imgs)
        image2 = cv2.cvtColor(np.float32(image2), cv2.COLOR_BGR2GRAY)
        image2 = np.sqrt(image2 / float(np.max(image2)))  # Square_root normalization
        image2 = image2 * 255
        hog_features = HOG.hog(image2)
        img_features.append(hog_features)

        return np.concatenate(img_features)
    else:
        features = []  # Creating an array to store features of all images
        i = 0
        for image in imgs:  # Iterating through all images
            image_features = []  # Creating an array to store
            img = np.copy(image)
            HEIGHT, WIDTH = img.shape[:2]

            # IMAGE AUGMENTATIONS
            if 500 < i < 1000 and i % 7 == 0:
                img = np.fliplr(img)  # Flip image

            elif 1000 < i < 1500 and i % 7 == 0:
                # Shifting Right
                M = np.float32([[1, 0, 20], [0, 1, 0]])
                img = cv2.warpAffine(img, M, (HEIGHT, WIDTH))

            elif 1500 < i < 2000 and i % 7 == 0:
                # Shifting Left
                M = np.float32([[1, 0, -20], [0, 1, 0]])
                img = cv2.warpAffine(img, M, (HEIGHT, WIDTH))

            elif 500 < i < 1500 and i % 7 == 0:
                img = np.fliplr(img)  # Flip image

            elif 2500 < i < 3000 and i % 6 == 0:
                # Shifting Up
                M = np.float32([[1, 0, 10], [0, 1, 20]])
                img = cv2.warpAffine(img, M, (HEIGHT, WIDTH))

            elif 3500 < i < 5000 and i % 7 == 0:
                # Shifting Down
                M = np.float32([[1, 0, 10], [0, 1, -25]])
                img = cv2.warpAffine(img, M, (HEIGHT, WIDTH))

            spatial_features = spatial.spatial(img, (32, 32))
            image_features.append(spatial_features)

            image2 = np.copy(img)
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2LUV)
            image2 = np.sqrt(image2)
            hist_features = ColorHist.color_histogram(image2)
            image_features.append(hist_features)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale to compute HOG
            img = np.sqrt(img / float(np.max(img)))  # Square_root normalization
            img = img * 255
            hog_features = HOG.hog(img)

            image_features.append(hog_features)
            features.append(np.concatenate(image_features))
            i += 1

    return features

##### Begin Feature Extraction #####
if __name__ == '__main__':

    vehicles_array = glob.glob('Vehicles\*.png')  # Load the positive image dataset
    vehicles = []
    for path in vehicles_array:
        image = cv2.imread(path)
        cvtimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vehicles.append(cvtimage)
    print("Loaded Vehicles Dataset\n")

    vehicles_array = glob.glob('Non-Vehicles\*.png')  # Load the negative image dataset
    no_vehicles = []
    for path in vehicles_array:
        image = cv2.imread(path)
        cvtimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        no_vehicles.append(cvtimage)
    print("Loaded Non-Vehicles Dataset\n")

    print("Length of vehicles dataset:", len(vehicles))
    print("Length of no_vehicles dataset:", len(no_vehicles))
    print("\n")

    vehicle_features = extract_features(vehicles[:3500], single_image=False)
    print("Extraction of All Vehicle Features Complete")
    print("Vehicle Features:", len(vehicle_features))
    savetxt('car_features1.csv', vehicle_features, delimiter=',')
    print('Vehicles Feature Vector Shape = ', np.array(vehicle_features).shape)

    notvehicle_features = extract_features(no_vehicles[:3500], single_image=False)
    print("Extraction of All Non-Vehicle Features Complete")
    print("Non Vehicle Features:", len(notvehicle_features))
    savetxt('notcar_features1.csv', notvehicle_features, delimiter=',')
    print('Non-Vehicles Feature Vector Shape = ', np.array(notvehicle_features).shape)

    ##### End Feature Extraction #####





    # Preparing the feature vector for training
    X = np.vstack((vehicle_features, notvehicle_features)).astype(np.float64)  # Stack both features vertically
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(notvehicle_features))))  # Create the labels vector

    # Normalizing the feature vectors
    scaler = StandardScaler()  # Load scaler to scale the feature vector
    X_scaler = scaler.fit(X)  # Per Column Scaler
    scaled_X = X_scaler.transform(X)  # Scaling the stacked feature vector X

    savetxt('Scaler1.csv', X, delimiter=',')

    rand_state = np.random.randint(0, 100)
    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.05)

    print("X_train Length", len(X_train))
    print("y_train Length", len(y_train))
    print("Shape of X_train", X_train.shape)
    print('Length of the feature vector:', len(X_train[0]))

    svc = SVC(C=100, kernel='linear', probability=True)
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...\n')

    print('Test Accuracy = ', round(svc.score(X_test, y_test), 4))

    preds1 = svc.predict_proba(X_test)


    preds = svc.predict(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)  # Create ROC curve

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    joblib.dump(svc, 'Classifier1.pkl')  # Save the classifier for later use
    print("TRAINED CLASSIFIER SAVED")
    print("FEATURE EXTRACTION COMPLETE")
    