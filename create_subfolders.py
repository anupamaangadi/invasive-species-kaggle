import os

import pandas as pd

DATA_PATH = '/media/hdd/training_data/invasive-species'
TRAIN_PATH = os.path.join(DATA_PATH, 'train')
VALID_PATH = os.path.join(DATA_PATH, 'validation')
LABELS_PATH = os.path.join(DATA_PATH, 'train_labels.csv')

assert os.path.exists(TRAIN_PATH)

if not os.path.exists(os.path.join(TRAIN_PATH, '0')):
    os.makedirs(os.path.join(TRAIN_PATH, '0'))

if not os.path.exists(os.path.join(TRAIN_PATH, '1')):
    os.makedirs(os.path.join(TRAIN_PATH, '1'))

if not os.path.exists(os.path.join(DATA_PATH, 'validation')):
    os.makedirs(os.path.join(DATA_PATH, 'validation'))

if not os.path.exists(os.path.join(VALID_PATH, '0')):
    os.makedirs(os.path.join(VALID_PATH, '0'))

if not os.path.exists(os.path.join(VALID_PATH, '1')):
    os.makedirs(os.path.join(VALID_PATH, '1'))

labels_df = pd.read_csv(LABELS_PATH)
idx = labels_df['name'].as_matrix()
label = labels_df['invasive'].as_matrix()

for id, invasive in zip(idx, label):
    os.rename(os.path.join(TRAIN_PATH, f'{id}.jpg'),
              os.path.join(TRAIN_PATH, str(invasive), f'{id}.jpg'))
