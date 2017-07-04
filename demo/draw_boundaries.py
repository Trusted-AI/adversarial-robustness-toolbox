import random
import sys

import numpy as np

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import keras.backend as K
import tensorflow as tf

from utils import *

M = 20 # nb instances of original dataset
H = .02  # step size in the mesh
EPOCHS = 100 # nb training epochs
EPS = 0.3 # scale of the perturbations

STRATEGIES = ["original", "uniform", "gaussian", "fgm", "vat", "jsma"]

r = np.random.RandomState(17)
random.seed(17)

session = tf.Session()
K.set_session(session)

# get dataset
dataset = sys.argv[1]
X, Y_labels, Y_cat = get_toyset(dataset, nb_instances=M, rnd_state=r)

# create a mesh to plot in
x_min,x_max = X[:, 0].min() - 0.2, X[:, 0].max() + 0.2
y_min, y_max = X[:, 1].min() - 0.2, X[:, 1].max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, H), np.arange(y_min, y_max, H))

# train original model
model_original = simple_nn()
model_original.compile(**{"loss": 'categorical_crossentropy', "optimizer": 'adam', "metrics": ['accuracy']})
model_original.fit(X, Y_cat, verbose=0, batch_size=M // 10, epochs=EPOCHS)

# Plot results
colors = [(1, 1, 1), (0.5, 0.6, 1)]
cm = LinearSegmentedColormap.from_list('twocolor', colors, N=100)

plt.figure()
reds = Y_labels == 1
blues = Y_labels == 0

for i,s in enumerate(STRATEGIES):

    if s == "original":
        model = model_original

    else:
        X_aug, Y_aug = data_augmentation(X, Y_cat, type=s, model=model_original, session=session)

        # data augmentation
        X_train = np.append(X, X_aug, axis=0)
        Y_train = np.append(Y_cat, Y_aug, axis=0)

        model = simple_nn()
        model.compile(**{"loss": 'categorical_crossentropy', "optimizer": 'adam', "metrics": ['accuracy']})
        model.fit(X_train, Y_train, verbose=0, batch_size=M // 10, epochs=EPOCHS)

    plt.subplot(2, 3, i+1, aspect='equal')
    plt.title("{} results".format(s))

    confs_grid = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # draw classification contours
    y_grid = np.argmax(confs_grid, axis=1).reshape(xx.shape)
    plt.contourf(xx, yy, y_grid, cmap=cm, alpha=0.3)

    # draw level contours
    y_grid = confs_grid[:, 0].reshape(xx.shape)
    plt.contour(xx, yy, y_grid, colors='grey', linewidths=1, origin='lower')

    plt.scatter(X[reds, 0], X[reds, 1], edgecolor="black", cmap=cm, s=60, marker="o", c=colors[1])
    plt.scatter(X[blues, 0], X[blues, 1], edgecolor="black", cmap=cm, s=60, marker="^", c=colors[0])

    if s != "original":
        plt.scatter(X_aug[reds, 0], X_aug[reds, 1], edgecolor="black", s=20, cmap=cm, marker="o", c=colors[1])
        plt.scatter(X_aug[blues, 0], X_aug[blues, 1], edgecolor="black", s=20, cmap=cm, marker="^", c=colors[0])

    train_acc = model.evaluate(X, Y_cat, verbose=0)
    plt.xlabel("train accuracy = %d%%" % (train_acc[1] * 100))

plt.subplots_adjust(0.02, 0.10, 0.98, 0.94, 0.45, 0.35)

plt.show()