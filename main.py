from tools import TRAIN_PATH, create_paths, load_labels, LABELS_PATH, load_images
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # print(create_paths(TRAIN_PATH))
    # invasive = load_labels(LABELS_PATH)
    # print(id)
    # print(invasive)
    image_paths = create_paths(TRAIN_PATH)
    images = load_images(image_paths, size=20)
    plt.imshow(images[0])
    plt.show()
