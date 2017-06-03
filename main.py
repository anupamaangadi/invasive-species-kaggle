import os

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

from tools import TRAIN_PATH, VALID_PATH

SAVE_PATH = '/media/hdd/saved-models/invasive-species/'
img_width, img_height = 150, 150
batch_size = 16
epochs = 50
nb_train_samples = 1869
nb_validation_samples = 426

if __name__ == '__main__':
    # invasive = load_labels(LABELS_PATH)
    # image_paths = create_paths(TRAIN_PATH)

    # all_images = load_all_images(image_paths)
    # labels_raw = np.asarray(list(invasive.values()))
    #
    # train_images = all_images[:2250]
    # train_labels = labels_raw[:2250]
    # valid_images = all_images[2250:]
    # valid_labels = labels_raw[2250:]

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=.2,
        zoom_range=.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_PATH,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        VALID_PATH,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary'
    )

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[ModelCheckpoint(
            filepath='/media/hdd/saved-models/invasive-species/checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5',
            save_best_only=True)])
