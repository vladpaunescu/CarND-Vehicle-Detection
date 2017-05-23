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
__C.HOG_CHANNEL = 0

# use this for opencv reading
__C.HOG_COLORSPACE = cv2.COLOR_BGR2YCrCb

# use this for video reading frame by frame
# __C.HOG_COLORSPACE = cv2.COLOR_BGRYCrCb


__C.USE_HOG = True
__C.USE_SPATIAL_BINNING = True
__C.USE_COLOR_HIST = True


cfg = __C


