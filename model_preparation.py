import random

import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from data_preparation import get_train_data, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
from function import mean_iou


USE_DATA_GEN = False


def model_Unet(height, width, channels):
    # Build U-Net model
    inputs = Input((height, width, channels))
    #s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()

    return model


def data_gen():

    args = dict(shear_range=0.5,
                rotation_range=360,
                zoom_range=0.2,
                width_shift_range=0.2,
                height_shift_range=0.2,
                fill_mode='reflect')

    gen1 = ImageDataGenerator(**args)
    gen2 = ImageDataGenerator(**args)

    return gen1, gen2


if __name__ == "__main__":

    X_train, Y_train, ids = get_train_data()

    model = model_Unet(IMG_HEIGHT, IMG_WIDTH, 1)

    # Fit model
    earlystopper = EarlyStopping(patience=3, verbose=1)
    checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)

    if not USE_DATA_GEN:
        model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
                  callbacks=[earlystopper, checkpointer])
    else:
        img_gen, mask_gen = data_gen()

        seed = 1

        img_gen.fit(X_train, augment=True, seed=seed)
        mask_gen.fit(Y_train, augment=True, seed=seed)

        image_generator = img_gen.flow(X_train[:int(X_train.shape[0]*0.9)], seed=seed, batch_size=16)
        mask_generator = mask_gen.flow(Y_train[:int(Y_train.shape[0]*0.9)], seed=seed, batch_size=16)

        train_generator = zip(image_generator, mask_generator)

        x_val = img_gen.flow(X_train[int(X_train.shape[0] * 0.9):], batch_size=16, shuffle=True,
                                       seed=seed)
        y_val = mask_gen.flow(Y_train[int(Y_train.shape[0] * 0.9):], batch_size=16, shuffle=True,
                                      seed=seed)

        val_generator = zip(x_val, y_val)

        model.fit_generator(train_generator, steps_per_epoch=60, epochs=50,validation_data=val_generator, validation_steps=5,
                            callbacks=[earlystopper, checkpointer])
