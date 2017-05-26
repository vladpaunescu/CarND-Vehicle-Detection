import glob
import numpy as np

from config import cfg


import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Define a function to return some characteristics of the dataset
def data_look(car_dir, notcar_dir):
    cars = glob.glob(car_dir + '/*/*.png')
    notcars = glob.glob(notcar_dir + '/*/*.png')
    print("Total cars examples {}".format(len(cars)))
    print("Total non cars examples {}".format(len(notcars)))

    return cars, notcars


if __name__ == "__main__":
    cars, notcars = data_look(cfg.CARS_DIR, cfg.NON_CARS_DIR)

    # plot 2 samples
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    # Read in car / not-car images
    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    # Plot the examples
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(car_image)
    plt.title('Car Image')
    plt.subplot(122)
    plt.imshow(notcar_image)
    plt.title('Not-car Image')

    plt.show()

