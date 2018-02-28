
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop


def detector(win_h=64, win_w=64, win_ch=1, final_activation='softmax'):

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu", input_shape=(win_h, win_w, win_ch)))
    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu"))
    model.add(Conv2D(32, (5, 5), padding="Same", activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation=final_activation))

    # Compile the model
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


if __name__ == "__main__":

    model = detector()
