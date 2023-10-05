import numpy as np
from numpy import savetxt
import cv2
import matplotlib.pyplot as plt
import math

###############    HISTOGRAM OF GRADIENTS    ####################

cell_size = 10  # Size of a cell
bin_size = 9  # Number of bins
angle_unit = 20
height = 64
width = 64
cell_gradient_vector = np.zeros((height // cell_size, width // cell_size, bin_size))  # Create the cell gradient vector


def get_gradient(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Obtain Gradients iclean the X direction
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Obtain Gradients in the Y direction
    magnitude = np.sqrt(gx ** 2 + gy ** 2)  # Obtain Gradient Magnitude
    gradient_angle = cv2.phase(gx, gy, angleInDegrees=True)  # Obtain Gradient angle

    return magnitude, abs(gradient_angle)


def cell_gradient(cell_magnitude, cell_angle):
    orientation_centers = [0] * bin_size
    hog = []

    # Iterate through the cell in x and y directions (each pixel)
    for k in range(cell_magnitude.shape[0]):
        for l in range(cell_magnitude.shape[1]):
            angle = 0
            gradient_strength = cell_magnitude[k][l]  # Gradient value for one pixel
            gradient_angle = cell_angle[k][l]  # Gradient angle for one pixel

            if (gradient_angle <= 180):  # Making angle values in the range 0 to 9
                angle = int(gradient_angle // 20)

            if (gradient_angle >= 180):  # Making angle values in the range 0 to 9
                angle = int(gradient_angle // 40)

            if (angle < 9):
                orientation_centers[angle] += (gradient_strength / 2)  # Add the gradient weight to the respective bins
                orientation_centers[angle - 1] += (
                            gradient_strength / 2)  # Add the gradient weight to the respective bins
                
    # Plot the histogram
    '''
    bin_labels = [str(i) for i in range(9)]  # Labels for the bins
    plt.bar(bin_labels, orientation_centers)  # Create a bar chart
    plt.xlabel('Bin')
    plt.ylabel('Magnitude')
    plt.title('Histogram of Gradients')
    plt.show(block = True)  # Display the chart
    '''
    return orientation_centers


def hog(image):
    mag, angle = get_gradient(image)

    # Obtain gradient angle and magnitude for one cell from whole image
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = mag[i * cell_size:(i + 1) * cell_size,
                             # Obtain magnitude vector for one cell from full magnitude
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = angle[i * cell_size:(i + 1) * cell_size,  # Obtain angle vector for one cell from full angle
                         j * cell_size:(j + 1) * cell_size]

            cell_gradient_vector[i][j] = cell_gradient(cell_magnitude,
                                                       cell_angle)  # Pass cell gradient vector for histogram

            

    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []  # A vector to store the blocks

            # Take 4 cells to form a block
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])

            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))  # Getting the magnitude of the final vector
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    return np.concatenate(hog_vector)
    print(np.array(hog_vector).shape)