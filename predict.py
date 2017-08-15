import os
import pandas as pd

import numpy as np
from keras.models import load_model

from tools import SAVE_PATH, TEST_PATH, load_all_images, create_paths

if __name__ == '__main__':
    test_paths = create_paths(TEST_PATH)
    images_dict = load_all_images(test_paths)
    images = np.asarray(list(images_dict.values()))
    ids = np.asarray(list(images_dict.keys()))
    print(len(images))
    model = load_model(os.path.join(SAVE_PATH, 'checkpoint-72-0.14.hdf5'))
    predictions = model.predict(images)
    predictions = predictions.flatten()

    df = pd.DataFrame({
        'name': ids,
        'invasive': predictions
    })

    df.to_csv('submission.csv', sep=',', index=False)
