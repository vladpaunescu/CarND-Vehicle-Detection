import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
import time
import pickle
import os


import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from config import cfg
from dataset import data_look


def transform_colorspace(img, cspace=cv2.COLOR_RGB2YCrCb):
    return cv2.cvtColor(img, cspace)


def bin_spatial(img, size=(32, 32), visualize=False):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    if not visualize:
        return np.hstack((color1, color2, color3))

    return np.hstack((color1, color2, color3)), cv2.merge((color1.reshape((32,32)),
                                                          color2.reshape((32,32)),
                                                          color3.reshape((32,32))))


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector

    return hist_features


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image

    # Otherwise call with one output
    features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
    return features


def get_features_for_image(img, hog_channel="ALL", use_hog=True, use_spatial_binning = True, use_color_hist = True):
    feat_img = transform_colorspace(img, cfg.HOG_COLORSPACE)
    features = []
    if use_hog:
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feat_img.shape[2]):
                hog_features.append(get_hog_features(feat_img[:, :, channel],
                                                     orient=cfg.ORIENT,
                                                     pix_per_cell=cfg.PIX_PER_CELL,
                                                     cell_per_block=cfg.CELL_PER_BLOCK,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feat_img[:, :, hog_channel],  orient=cfg.ORIENT,
                                                     pix_per_cell=cfg.PIX_PER_CELL,
                                                     cell_per_block=cfg.CELL_PER_BLOCK, vis=False, feature_vec=True)

        features.append(hog_features)

    if use_spatial_binning:
        spatial_features = bin_spatial(feat_img, size=cfg.SPATIAL_SIZE)
        features.append(spatial_features)
    if use_color_hist:
        # Apply color_hist()
        hist_features = color_hist(feat_img, nbins=cfg.HIST_BINS)
        features.append(hist_features)

    return np.concatenate(features)

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs):

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        feature_descriptor = get_features_for_image(image,
                                                    hog_channel=cfg.HOG_CHANNEL,
                                                    use_hog=cfg.USE_HOG,
                                                    use_spatial_binning=cfg.USE_SPATIAL_BINNING,
                                                    use_color_hist=cfg.USE_COLOR_HIST)

        # Append the new feature vector to the features list
        features.append(feature_descriptor)

    # Return list of feature vectors

    return features

def visualize_sample_hog(img, hog_channel=0):
    feat_img = transform_colorspace(img, cfg.HOG_COLORSPACE)

    hog_features, hog_image = get_hog_features(feat_img[:, :, hog_channel], orient=cfg.ORIENT,
                                    pix_per_cell=cfg.PIX_PER_CELL,
                                    cell_per_block=cfg.CELL_PER_BLOCK, vis=True, feature_vec=True)

    return feat_img[:, :, hog_channel], hog_image


if __name__ == "__main__":
    # cars, notcars = data_look(cfg.CARS_DIR, cfg.NON_CARS_DIR)
    # t = time.time()
    # car_features = extract_features(cars)
    # notcar_features = extract_features(notcars)
    # t2 = time.time()
    # print(round(t2 - t, 2), 'Seconds to extract HOG features...')

    cars, notcars = data_look(cfg.CARS_DIR, cfg.NON_CARS_DIR)
    # plot 2 samples
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = cv2.imread(cars[car_ind])
    notcar_image = cv2.imread(notcars[notcar_ind])

# hog printing

    channels_count = 3
    colorspace = ['Y', 'Cr', 'Cb']

    # for ch in range(channels_count):
    #     # feat_img, hog_image = visualize_sample_hog(car_image, hog_channel=ch)
    #     feat_img, hog_image = visualize_sample_hog(notcar_image, hog_channel=ch)
    #     # Plot the examples
    #     fig = plt.figure()
    #     plt.subplot(121)
    #     plt.imshow(feat_img)
    #     plt.title('Not Car Image. {} channel'.format(colorspace[ch]))
    #     plt.subplot(122)
    #     plt.imshow(hog_image)
    #     plt.title('Not Car HOG. {} channel'.format(colorspace[ch]))
    #
    #     plt.show()

# spatial binning printing

    # spatial_binned_feats, spatial_binned_img = bin_spatial(car_image, size=(32, 32), visualize=True)
    # fig = plt.figure()
    # plt.subplot(121)
    # plt.title('Car Image')
    # plt.imshow(car_image)
    # plt.subplot(122)
    # plt.title('Spatial Binned Car Image')
    # plt.imshow(spatial_binned_img)
    # plt.show()

# color histogram printing
    car_imgycrcb = transform_colorspace(notcar_image, cspace=cv2.COLOR_BGR2YCrCb)
    bin_size = int(256/32)
    for ch in range(channels_count):
        fig = plt.figure()
        plt.subplot(121)
        plt.title('Not Car {} channel'.format(colorspace[ch]))
        plt.imshow(car_imgycrcb[:, :, ch])
        plt.subplot(122)
        plt.title('Not car {} histogram'.format(colorspace[ch]))
        plt.hist(car_imgycrcb[:, :, ch], bins=range(0, 256, bin_size))

        plt.show()
