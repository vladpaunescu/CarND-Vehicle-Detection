""" config file """

from os.path import join
import cv2

class Property(object): pass

__C = Property()

# dataset properties

__C.DATASET_DIR = "./dataset"
__C.CARS_DIR = join(__C.DATASET_DIR, "vehicles")
__C.NON_CARS_DIR = join(__C.DATASET_DIR, "non-vehicles")

# HOG config
__C.PIX_PER_CELL = 8
__C.CELL_PER_BLOCK = 2
__C.ORIENT = 9
__C.HOG_CHANNEL = "ALL"

# use this for opencv reading
# __C.HOG_COLORSPACE = cv2.COLOR_BGR2YCrCb

# use this for video reading frame by frame
__C.HOG_COLORSPACE = cv2.COLOR_RGB2YCrCb

__C.SPATIAL_SIZE = (32, 32) # Spatial binning dimensions
__C.HIST_BINS = 32    # Number of histogram bins
__C.HIST_RANGE = (0, 256)


__C.USE_HOG = True
__C.USE_SPATIAL_BINNING = True
__C.USE_COLOR_HIST = True

# binary filenames
__C.FEATURES_BIN = "./features.pkl"
__C.MODEL_BIN = "./model.pkl"

__C.TEST_IMG_DIR = "./test_images"


# moving window search config
__C.X_START_STOP = [640, None]
__C.Y_START_STOP = [None, None] # Min and max in y to search in slide_window()

__C.Y_START_STOP_FAR = [380, 550]
__C.Y_START_STOP_NEAR = [380, 650]


# detection threshold
__C.DET_THRESHOLD = 4
__C.OVERLAP = 0.9

__C.XY_WINDOW_FAR = (120, 96)
__C.XY_WINDOW_NEAR = (280, 224)


# test dirs
__C.TEST_DIR = "./test_images"
__C.OUT_DIR = "./detection_test_images"

__C.IMG_EXT = ".png"


# video options
__C.TEST_VIDEO = "./project_video.mp4"
__C.OUT_VIDEO = "./detections_video_test22.mp4"


cfg = __C


