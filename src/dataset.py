import glob

from config import cfg

images = glob.glob('*.jpeg')
cars = []
notcars = []

for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)


# Define a function to return some characteristics of the dataset
def data_look(car_dir, notcar_dir):
    cars = glob.glob(car_dir + '/*/*.png')
    notcars = glob.glob(notcar_dir + '/*/*.png')
    print("Total cars examples {}".format(len(cars)))
    print("Total non cars examples {}".format(len(notcars)))

    return cars, notcars


if __name__ == "__main__":

    cars, notcars = ata_look(cfg.CARS_DIR, cfg.NON_CARS_DIR)