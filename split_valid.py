import os

import numpy as np

from tools import TRAIN_PATH, VALID_PATH

SPLIT_RATIO = .2

invasive_len = len(os.listdir(os.path.join(TRAIN_PATH, '1')))
noninvasive_len = len(os.listdir(os.path.join(TRAIN_PATH, '0')))

print(invasive_len)
print(noninvasive_len)

valid_len_invasive = int(invasive_len * SPLIT_RATIO)
valid_len_noninvasive = int(noninvasive_len * SPLIT_RATIO)

invasive_files = os.listdir(os.path.join(TRAIN_PATH, '1'))
noninvasive_files = os.listdir(os.path.join(TRAIN_PATH, '0'))

invasive_files = np.random.permutation(invasive_files)[:valid_len_invasive]
noninvasive_files = np.random.permutation(noninvasive_files)[
                    :valid_len_noninvasive]

for invasive_file in invasive_files:
    os.rename(os.path.join(TRAIN_PATH, '1', invasive_file),
              os.path.join(VALID_PATH, '1', invasive_file))

for noninvasive_file in noninvasive_files:
    os.rename(os.path.join(TRAIN_PATH, '0', noninvasive_file),
              os.path.join(VALID_PATH, '0', noninvasive_file))
