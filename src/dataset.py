import glob

from config import cfg


# Define a function to return some characteristics of the dataset
def data_look(car_dir, notcar_dir):
    cars = glob.glob(car_dir + '/*/*.png')
    notcars = glob.glob(notcar_dir + '/*/*.png')
    print("Total cars examples {}".format(len(cars)))
    print("Total non cars examples {}".format(len(notcars)))

    return cars, notcars


if __name__ == "__main__":
    cars, notcars = data_look(cfg.CARS_DIR, cfg.NON_CARS_DIR)