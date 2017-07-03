import random

import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
import tensorflow as tf

from utils import *

M = 20 # nb instances of original dataset
H = .02  # step size in the mesh
EPOCHS = 100 # nb training epochs
EPS = 0.3 # scale of the perturbations

STRATEGIES = ["original", "uniform", "gaussian", "fgm", "vat"]

r = np.random.RandomState(5)
random.seed(5)

session = tf.Session()
K.set_session(session)

# get dataset
X, Y_labels, Y_cat = get_toyset("circles", nb_instances=M, rnd_state=r)

# create a mesh to plot in
x_min,x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))

# Plot results
plt.figure()
reds = Y_labels == 1
blues = Y_labels == 0

for i,s in enumerate(STRATEGIES):

    if s == "original":
        X_train = X
        Y_train = Y_cat

    else:
        X_aug, Y_aug = data_augmentation(X, Y_cat, type=s, model=model, session=session)

        # data augmentation
        X_train = np.append(X, X_aug, axis=0)
        Y_train = np.append(Y_cat, Y_aug, axis=0)

    # train model
    model = simple_nn()
    model.compile(**{"loss": 'categorical_crossentropy', "optimizer": 'adam', "metrics": ['accuracy']})
    model.fit(X_train, Y_train, verbose=0, batch_size=M // 10, epochs=EPOCHS)

    plt.subplot(2, 3, i+1, aspect='equal')
    plt.title("{} results".format(s))

    plt.plot(X[reds, 0], X[reds, 1], "ro")
    plt.plot(X[blues, 0], X[blues, 1], "bo")
    # plt.xlabel("$x_1$")
    # plt.ylabel("$x_2$")

    confs_grid = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # draw classification contours
    y_grid = np.argmax(confs_grid, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, y_grid, cmap=plt.cm.coolwarm, alpha=0.3)

    # draw level contours
    y_grid = confs_grid[:, 0].reshape(xx.shape)
    plt.contour(xx, yy, y_grid, colors='grey', linewidths=1, origin='lower')

plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.45, 0.35)

plt.show()