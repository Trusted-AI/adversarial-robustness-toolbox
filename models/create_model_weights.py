

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

import numpy as np

from art.utils import load_dataset, master_seed

def main_mnist_binary():

    master_seed(1234)

    model = Sequential()
    model.add(Conv2D(1, kernel_size=(7, 7), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.01), metrics=['accuracy'])

    (x_train, y_train), (_, _), _, _ = load_dataset('mnist')

    y_train = np.argmax(y_train, axis=1)
    y_train[y_train < 5] = 0
    y_train[y_train >= 5] = 1

    model.fit(x_train, y_train, batch_size=128, epochs=10)

    w_0, b_0 = model.layers[0].get_weights()
    w_3, b_3 = model.layers[3].get_weights()

    np.save('W_CONV2D_MNIST_BINARY', w_0)
    np.save('B_CONV2D_MNIST_BINARY', b_0)
    np.save('W_DENSE_MNIST_BINARY', w_3)
    np.save('B_DENSE_MNIST_BINARY', b_3)


if __name__ == '__main__':
    main_mnist_binary()
