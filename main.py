from tools import TRAIN_PATH, create_paths, load_labels, LABELS_PATH

if __name__ == '__main__':
    # print(create_paths(TRAIN_PATH))
    invasive = load_labels(LABELS_PATH)
    # print(id)
    print(invasive)