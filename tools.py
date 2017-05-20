import os
import random
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


def load_images(file_paths, size=0):
    images = []

    if size == 0:
        size = len(file_paths)
    mask = random.sample(range(len(file_paths)), k=size)
    for i in mask:
        images.append(np.array(Image.open(file_paths[i])))
    return images
