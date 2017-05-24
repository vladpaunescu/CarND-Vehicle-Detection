import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle
import os

from config import cfg
from dataset import data_look
from features import extract_features


def train_model(car_features, notcar_features):
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return svc, X_scaler

def get_features():
    if os.path.exists(cfg.FEATURES_BIN):
        with open(cfg.FEATURES_BIN, 'rb') as f:
            features = pickle.load(f)
            print("Read features from disk")

            car_features = features['car_features']
            notcar_features = features['notcar_features']
        return car_features, notcar_features

    # no cached features
    print("dataset reading")
    cars, notcars = data_look(cfg.CARS_DIR, cfg.NON_CARS_DIR)
    print("DONE dataset reading")

    t = time.time()
    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to extract HOG features...')

    features = {'car_features': car_features,
             'notcar_features': notcar_features}

    # Save the features on disk
    with open(cfg.FEATURES_BIN, 'wb') as f:
        pickle.dump(features, f)
        print("Dumped the features on disk")

    return car_features, notcar_features

if __name__ == "__main__":

    car_features, notcar_features = get_features()

    svc, X_scaler = train_model(car_features, notcar_features)

    model = {'svc': svc,
             'X_scaler': X_scaler }

    # Save the model on disk
    with open(cfg.MODEL_BIN, 'wb') as f:
        pickle.dump(model, f)