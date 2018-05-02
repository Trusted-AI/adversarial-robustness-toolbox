import numpy as np
import os
import random
import sys

import matplotlib.pyplot as plt

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils

from sklearn.datasets import make_moons, make_circles, make_swiss_roll
from sklearn.preprocessing import MinMaxScaler

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from art.attackers.fast_gradient import FastGradientMethod
from art.attackers.saliency_map import SaliencyMapMethod
from art.attackers.virtual_adversarial import VirtualAdversarialMethod
from art.utils import make_directory

# -------------------------------------------------------------------------------------------------------- IO FUNCTIONS

def save_fig(filename,fig_id=None,output_dir="pics/",format="pdf"):

    if fig_id:
        plt.figure(fig_id)

    make_directory(output_dir)

    plt.savefig(os.path.join(output_dir, filename+"."+format))

# -------------------------------------------------------------------------------------------------------- TOY DATASETS

def get_toyset(name, nb_instances=20, noise=None, factor=0.6, rnd_state=None):

    if name == "moons":
        X, Y = make_moons(n_samples=nb_instances*10, noise=noise, random_state=rnd_state)

    elif name == "circles":
        X, Y = make_circles(n_samples=nb_instances*10, noise=noise, factor=factor, random_state=rnd_state)

    elif name == "swissroll":

        X1,_ = make_swiss_roll(n_samples=nb_instances*5, noise=0)
        X1 = X1[:, ::2]
        Y1 = np.ones((nb_instances*5,))

        X2 = np.random.uniform([X1[:, 0].min(), X1[:, 1].min()], high=[X1[:, 0].max(), X1[:, 1].max()],
                               size=(nb_instances*5, 2))
        Y2 = np.zeros((nb_instances*5,))

        X = np.r_[X1, X2]
        Y = np.r_[Y1, Y2]

    else:
        raise NotImplementedError("unknown dataset")

    indices = random.sample(range(nb_instances*10), nb_instances)
    X, Y = X[indices], Y[indices]

    X = MinMaxScaler((-1, 1)).fit_transform(X)
    y_cat = np_utils.to_categorical(Y, 2)

    return X, Y, y_cat

# ------------------------------------------------------------------------------------------------------ LEARNING UTILS

def simple_nn(nb_units=64):
    # as first layer in a sequential model:
    model = Sequential()
    model.add(Dense(nb_units, input_shape=(2,)))
    # model.add(Dropout(0.1))
    model.add(Dense(nb_units, activation="relu"))
    # model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))

    return model

def data_augmentation(x, y, type="gaussian", nb_instances=10, eps=0.3, **kwargs):

    if type == "uniform":
        x_aug = np.random.uniform(low=[-eps, -eps], high=[eps, eps], size=(nb_instances,) + x.shape)
        x_aug = np.tile(x, (nb_instances, 1)) + x_aug.reshape(-1, *x[0].shape)
        y_aug = np.tile(y, (nb_instances, 1))

    elif type == "gaussian":
        x_aug = np.random.normal(x, scale=eps, size=(nb_instances,) + x.shape)
        x_aug = x_aug.reshape(-1, *x[0].shape)
        y_aug = np.tile(y, (nb_instances, 1))

    elif type == "fgm":
        adv_crafter = FastGradientMethod(model=kwargs["model"], sess=kwargs["session"], ord=2)
        x_aug = adv_crafter.generate(x=x, eps=eps)
        y_aug = y.copy()

    elif type == "vat":
        adv_crafter = VirtualAdversarialMethod(model=kwargs["model"], sess=kwargs["session"])
        x_aug = adv_crafter.generate(x=x, eps=eps)
        y_aug = y.copy()

    elif type == "jsma":
        adv_crafter = SaliencyMapMethod(model=kwargs["model"], sess=kwargs["session"], gamma=1., theta=0.1)
        x_aug = adv_crafter.generate(x=x, eps=eps, nb_classes=2)
        y_aug = y.copy()

    x_aug = np.clip(x_aug, (x[:, 0].min(), x[:, 1].min()), (x[:, 0].max(), x[:, 1].max()))

    return x_aug, y_aug