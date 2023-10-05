import numpy as np
from numpy import savetxt
import cv2
import matplotlib.pyplot as plt

###############    COLOR HISTOGRAM ####################

hist_bins = 32  # Number of histogram bins


def color_histogram(img):
    nbins = 32
    bins_range = (0, 256)
    channel1 = img[:, :, 0]
    channel2 = img[:, :, 1]
    channel3 = img[:, :, 2]
    # Considering each channel seperately
    channel1_hist = np.histogram(channel1, bins=nbins, range=bins_range)
    channel2_hist = np.histogram(channel2, bins=nbins, range=bins_range)
    channel3_hist = np.histogram(channel3, bins=nbins, range=bins_range)
    hist_features = np.concatenate(
        (channel1_hist[0], channel2_hist[0], channel3_hist[0]))  # Obtain final color features
    return hist_features


###############   END OF COLOR HISTOGRAM ####################