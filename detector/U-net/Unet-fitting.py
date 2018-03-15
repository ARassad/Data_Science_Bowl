import random

import numpy as np

from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
#from data_preparation import get_train_data, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS
from function import mean_iou
#import model_preparation as mp
import os
from tqdm import tqdm
from Data import get_data


def model_Unet(height, width, channels):
    # Build U-Net model
    inputs = Input((height, width, channels))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(p2)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same')(c5)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c5)
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


if __name__ == "__main__":
    h, w, ch = 32, 32, 1
    size = None

    X_train, Y_train = get_data("../../../data/detector/", length=size, size=(h, w, ch))

    model = model_Unet(h, w, ch)

    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint('Unet(32x32).h5', verbose=1, save_best_only=True)

    model.fit(X_train, Y_train, validation_split=0.1, batch_size=16, epochs=50,
          callbacks=[earlystopper, checkpointer])
