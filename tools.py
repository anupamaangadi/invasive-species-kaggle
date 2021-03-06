import os
import random

import cv2
import numpy as np
import pandas as pd

PATH = '/media/hdd/training_data/invasive-species/'

# PATH = '/home/kamil/Dokumenty/invasive-species/'
TRAIN_PATH = os.path.join(PATH, 'train')
VALID_PATH = os.path.join(PATH, 'validation')
SAVE_PATH = '/media/hdd/saved-models/invasive-species'
TEST_PATH = os.path.join(PATH, 'test')
LABELS_PATH = os.path.join(PATH, 'train_labels.csv')
img_width, img_height = 224, 224


def create_paths(path):
    """This function creates paths to files contained in \path"""
    files = [os.path.join(path, file) for file in os.listdir(path)]
    return files


def load_labels(path):
    labels = pd.read_csv(path)
    return dict(zip(np.array(labels['name']), np.array(labels['invasive'])))


def load_all_images(file_paths):
    return {file_path.split('/')[-1].split('.')[0]: normalize(
        resize(cv2.imread(file_path))) for file_path in
        file_paths}


def next_batch(images, labels, grayscale=True, size=20):
    mask = random.sample(range(len(images)), k=size)
    return images[mask], labels[mask]


def normalize(images):
    return np.asarray([image / 255. for image in images])


def resize(image, size=(img_height, img_width)):
    return cv2.resize(image, size)
