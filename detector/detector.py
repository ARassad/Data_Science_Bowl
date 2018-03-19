
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from detector_data_prep import PATH_TO_NON_NUCL, PATH_TO, get_nucleas
import numpy as np
from keras import applications


def detector(win_h=22, win_w=22, win_ch=1, final_activation='sigmoid'):

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu", input_shape=(win_h, win_w, win_ch)))
    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu"))
    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation=final_activation))

    # Compile the model
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def detector_mobilenet(win_h=32, win_w=32, win_ch=1, final_activation='sigmoid'):

    initial_model = applications.MobileNet(include_top=False, input_shape=(win_h, win_w, win_ch), weights=None)

    x = initial_model.output
    x = Flatten()(x)
    out = Dense(1, activation=final_activation)(x)

    detector = Model(inputs=initial_model.input, outputs=out)
    detector.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    detector.summary()

    return detector, initial_model


if __name__ == "__main__":

    size = None
    h, w, ch = 32, 32, 1

    X_train, _ = get_nucleas(size, dir=PATH_TO, only_image=True, shape=(h, w, ch), as_grey=True)
    Y_train = np.array([1] * len(X_train))

    X_train_N, _ = get_nucleas(size, dir="../../data/detector_non_second/", only_image=True, shape=(h, w, ch), as_grey=True)
    Y_train_N = np.array([0] * len(X_train_N))

    X = np.concatenate((X_train, X_train_N))
    Y = np.concatenate((Y_train, Y_train_N))

    # Fit model
    earlystopper = EarlyStopping(patience=10, verbose=1)

    if False:
        checkpointer = ModelCheckpoint('detector(24x24).h5', verbose=1, save_best_only=True)

        model = detector(h, w)
        model.fit(X, Y, validation_split=0.1, batch_size=16, epochs=50, shuffle=True,
                  callbacks=[earlystopper, checkpointer])
    elif True:
        checkpointer = ModelCheckpoint('detector_MobileNet_Second.h5', verbose=1, save_best_only=True)

        detect, _ = detector_mobilenet(h, w, ch)

        detect.fit(X, Y, validation_split=0.1, batch_size=16, epochs=50, shuffle=True,
                   callbacks=[earlystopper, checkpointer])


