import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import os
from scipy.ndimage.measurements import label

from config import cfg
from features import get_features_for_image

y_start_stop = [None, None] # Min and max in y to search in slide_window()


DEBUG = False

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = get_features_for_image(test_img)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def detect(image, svc, X_scaler):
    draw_image = np.copy(image)

    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255


    windows = []

    windows += slide_window(image, x_start_stop=cfg.X_START_STOP,
                            y_start_stop=cfg.Y_START_STOP_FAR, xy_window=cfg.XY_WINDOW_FAR,
                            xy_overlap=(cfg.OVERLAP, cfg.OVERLAP))

    windows += slide_window(image, x_start_stop=cfg.X_START_STOP,
                            y_start_stop=cfg.Y_START_STOP_NEAR, xy_window=cfg.XY_WINDOW_NEAR,
                            xy_overlap=(cfg.OVERLAP, cfg.OVERLAP))

    hot_windows = search_windows(image, windows, svc, X_scaler)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, cfg.DET_THRESHOLD)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    print(labels[1], 'cars found')

    # plt.imshow(labels[0], cmap='gray')

    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    det_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    # window_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)

    # fig = plt.figure()
    # plt.subplot(121)
    # img_plot = cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_plot)
    # plt.title('Car Positions')
    # plt.subplot(122)
    # plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')

    # plt.imshow(labels[0], cmap='gray')
    # plt.title('Labels')
    # fig.tight_layout()

    # cv2.imwrite("search_windows_all.png", window_img)
    # cv2.imwrite("detections.png", det_img)

    return draw_img

def load_model():
    # load the model from disk
    with open(cfg.MODEL_BIN, 'rb') as f:
        model = pickle.load(f)

    return model['svc'], model['X_scaler']

# img_fname = "bbox-example-image.jpg"
img_fname = ["test1.jpg"]

if __name__ == "__main__":

    # plt.interactive(False)
    svc, X_scaler = load_model()

    imgs = os.listdir(cfg.TEST_DIR)
    print("Loaded test images {}".format(imgs))

    for img_path in imgs:
        img = cv2.imread(os.path.join(cfg.TEST_DIR, img_path))
        detect(image=img, svc=svc, X_scaler=X_scaler)
        plt.show()


