# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
from keras.models import load_model

from art.config import ART_DATA_PATH
from art.utils import load_dataset, get_file
from art.evaluations.security_curve.security_curve import SecurityCurve

from art.estimators.classification.keras import KerasClassifier


def main():
    kwargs = {
        "norm": "inf",
        "eps_step": 0.1,
        "max_iter": 3,
        "targeted": False,
        "num_random_init": 0,
        "batch_size": 32,
        "random_eps": False,
        "verbose": True,
    }

    (_, _), (x_test, y_test), min_, max_ = load_dataset("mnist")

    sc = SecurityCurve(eps=4)

    path = get_file(
        "mnist_cnn_robust.h5",
        extract=False,
        path=ART_DATA_PATH,
        url="https://www.dropbox.com/s/yutsncaniiy5uy8/mnist_cnn_robust.h5?dl=1",
    )
    robust_classifier_model = load_model(path)
    robust_classifier = KerasClassifier(clip_values=(min_, max_), model=robust_classifier_model, use_logits=False)

    sc.evaluate(classifier=robust_classifier, x=x_test, y=y_test, **kwargs)

    sc.plot()

    print(sc)


if __name__ == "__main__":
    main()
