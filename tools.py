import os
import random

import cv2
import pandas as pd
import numpy as np
from PIL import Image

# PATH = '/media/hdd/training_data/invasive-species/'
from keras.preprocessing.image import img_to_array, load_img

PATH = '/home/kamil/Dokumenty/invasive-species/'
TRAIN_PATH = os.path.join(PATH, 'train')
VALID_PATH = os.path.join(PATH, 'validation')
TEST_PATH = os.path.join(PATH, 'test')
LABELS_PATH = os.path.join(PATH, 'train_labels.csv')


def create_paths(path):
    """This function creates paths to files contained in \path"""
    files = [os.path.join(path, file) for file in os.listdir(path)]
    return files


def load_labels(path):
    labels = pd.read_csv(path)
    return dict(zip(np.array(labels['name']), np.array(labels['invasive'])))


def load_all_images(file_paths):
    # images = [load_img(file_path) for file_path in file_paths]
    return np.asarray([normalize(resize(img_to_array(cv2.imread(file_path)))) for file_path in file_paths])


def next_batch(images, labels, grayscale=True, size=20):
    mask = random.sample(range(len(images)), k=size)
    return images[mask], labels[mask]


def normalize(images):
    return np.asarray([image / 255. for image in images])


def resize(image, size=(150, 150)):
    return cv2.resize(image, size)
