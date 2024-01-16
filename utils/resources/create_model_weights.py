# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import numpy as np

from tests.utils import load_dataset, master_seed


def main_mnist_binary():
    master_seed(1234)

    model = Sequential()
    model.add(Conv2D(1, kernel_size=(7, 7), activation="relu", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=["accuracy"])

    (x_train, y_train), (_, _), _, _ = load_dataset("mnist")

    y_train = np.argmax(y_train, axis=1)
    y_train[y_train < 5] = 0
    y_train[y_train >= 5] = 1

    model.fit(x_train, y_train, batch_size=128, epochs=10)

    w_0, b_0 = model.layers[0].get_weights()
    w_3, b_3 = model.layers[3].get_weights()

    np.save(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "utils/resources/models/scikit/", "W_CONV2D_MNIST_BINARY"
        ),
        w_0,
    )
    np.save(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "utils/resources/models/scikit/" "B_CONV2D_MNIST_BINARY"
        ),
        b_0,
    )
    np.save(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "utils/resources/models/scikit/" "W_DENSE_MNIST_BINARY"
        ),
        w_3,
    )
    np.save(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "utils/resources/models/scikit/" "B_DENSE_MNIST_BINARY"
        ),
        b_3,
    )


def main_diabetes():
    master_seed(1234)

    model = Sequential()
    model.add(Dense(100, activation="relu", input_dim=10))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(lr=0.01), metrics=["accuracy"])

    (x_train, y_train), (_, _), _, _ = load_dataset("diabetes")

    model.fit(x_train, y_train, batch_size=128, epochs=10)

    w_0, b_0 = model.layers[0].get_weights()
    w_1, b_1 = model.layers[1].get_weights()
    w_2, b_2 = model.layers[2].get_weights()

    np.save(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models/", "W_DENSE1_DIABETES"),
        w_0,
    )
    np.save(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models/" "B_DENSE1_DIABETES"),
        b_0,
    )
    np.save(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models/" "W_DENSE2_DIABETES"),
        w_1,
    )
    np.save(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models/" "B_DENSE2_DIABETES"),
        b_1,
    )
    np.save(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models/" "W_DENSE3_DIABETES"),
        w_2,
    )
    np.save(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources/models/" "B_DENSE3_DIABETES"),
        b_2,
    )


def create_scikit_model_weights():
    master_seed(1234)

    model_list = {
        "decisionTreeClassifier": DecisionTreeClassifier(),
        "extraTreeClassifier": ExtraTreeClassifier(),
        "adaBoostClassifier": AdaBoostClassifier(),
        "baggingClassifier": BaggingClassifier(),
        "extraTreesClassifier": ExtraTreesClassifier(n_estimators=10),
        "gradientBoostingClassifier": GradientBoostingClassifier(n_estimators=10),
        "randomForestClassifier": RandomForestClassifier(n_estimators=10),
        "logisticRegression": LogisticRegression(solver="lbfgs", multi_class="auto"),
        "svc": SVC(gamma="auto"),
        "linearSVC": LinearSVC(),
    }

    clipped_models = {model_name: model for model_name, model in model_list.items()}
    unclipped_models = {model_name: model for model_name, model in model_list.items()}

    (x_train_iris, y_train_iris), (_, _), _, _ = load_dataset("iris")

    y_train_iris = np.argmax(y_train_iris, axis=1)

    for model_name, model in clipped_models.items():
        model.fit(X=x_train_iris, y=y_train_iris)
        pickle.dump(
            model,
            open(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "resources/models/scikit/",
                    "scikit-" + model_name + "-iris-clipped-ge-1.3.0.pickle",
                ),
                "wb",
            ),
        )

    for model_name, model in unclipped_models.items():
        model.fit(X=x_train_iris, y=y_train_iris)
        pickle.dump(
            model,
            open(
                os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    "resources/models/scikit/",
                    "scikit-" + model_name + "-iris-unclipped-ge-1.3.0.pickle",
                ),
                "wb",
            ),
        )


if __name__ == "__main__":
    # main_mnist_binary()
    create_scikit_model_weights()
    # main_diabetes()
