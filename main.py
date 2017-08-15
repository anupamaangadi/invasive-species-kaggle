import os

from keras import backend as K
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.engine import Model
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from tools import TRAIN_PATH, VALID_PATH, img_width, img_height, SAVE_PATH

batch_size = 16
epochs = 50
nb_train_samples = 1837
nb_validation_samples = 458

if __name__ == '__main__':
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
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        VALID_PATH,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary'
    )

    base_model = VGG16(include_top=False, input_shape=input_shape)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(.5))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Model(input=base_model.input, output=top_model(base_model.output))

    for layer in model.layers[:19]:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir=os.path.join(SAVE_PATH, 'logs'),
                              histogram_freq=1,
                              write_graph=True,
                              write_images=False)

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=2000,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[ModelCheckpoint(
            filepath=os.path.join(SAVE_PATH,
                                  'checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5'),
            save_best_only=True), tensorboard])
