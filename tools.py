import os
import random

import cv2
import pandas as pd
import numpy as np
from PIL import Image

PATH = '/media/hdd/training_data/invasive-species/'
TRAIN_PATH = os.path.join(PATH, 'train')
TEST_PATH = os.path.join(PATH, 'test')
LABELS_PATH = os.path.join(PATH, 'train_labels.csv')


def create_paths(path):
    """This function creates paths to files contained in \path"""
    files = [os.path.join(path, file) for file in os.listdir(path)]
    return files


def load_labels(path):
    labels = pd.read_csv(path)
    return dict(zip(np.array(labels['name']), np.array(labels['invasive'])))


def next_batch(file_paths, labels, grayscale=True, size=0):
    images = []

    if size == 0:
        size = len(file_paths)

    mask = random.sample(range(len(file_paths)), k=size)
    file_paths = np.asarray(file_paths)
    file_paths = file_paths[mask]

    chosen_labels = []
    filenames = [int(filename.split('/')[-1].split('.')[-2]) for filename in file_paths]
    for filename in filenames:
        chosen_labels.append(labels[filename])

    for file_path in file_paths:
        images.append(np.array(resize(cv2.imread(file_path))))

    if grayscale:
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    return normalize(images), np.array(chosen_labels)


def normalize(images):
    return np.array([image / 255. for image in images])


def resize(image, size=(224, 224)):
    return cv2.resize(image, size)
