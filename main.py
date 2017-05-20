import cv2

from tools import TRAIN_PATH, create_paths, load_labels, LABELS_PATH, load_images, normalize
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # print(create_paths(TRAIN_PATH))
    # invasive = load_labels(LABELS_PATH)
    # print(id)
    # print(invasive)
    image_paths = create_paths(TRAIN_PATH)
    images = load_images(image_paths, grayscale=True, size=20)
    # normalize(images)
    cv2.imshow('gray_image', images[0])
    cv2.waitKey(0)
    # plt.show()
