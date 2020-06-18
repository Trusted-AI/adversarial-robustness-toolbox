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
import os
import logging
import numpy as np
import pytest
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from art.estimators.classification.pytorch import PyTorchClassifier

from tests.utils import ExpectedValue

from tests.classifiersFrameworks.utils import (
    backend_test_fit_generator,
    backend_test_class_gradient,
    backend_test_loss_gradient,
)

logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(1, 2, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(288, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, 288)
        logit_output = self.fc(x)
        return logit_output


@pytest.mark.only_with_platform("pytorch")
def test_device():
    class Flatten(nn.Module):
        def forward(self, x):
            n, _, _, _ = x.size()
            result = x.view(n, -1)

            return result

    # Define the network
    model = nn.Sequential(nn.Conv2d(1, 2, 5), nn.ReLU(), nn.MaxPool2d(2, 2), Flatten(), nn.Linear(288, 10))

    # Define a loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # First test cpu
    classifier_cpu = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=(1, 28, 28),
        nb_classes=10,
        device_type="cpu",
    )

    assert classifier_cpu._device == torch.device("cpu")
    assert classifier_cpu._device != torch.device("cuda")

    # Then test gpu
    if torch.cuda.device_count() >= 2:
        with torch.cuda.device(0):
            classifier_gpu0 = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=loss_fn,
                optimizer=optimizer,
                input_shape=(1, 28, 28),
                nb_classes=10,
            )
            assert classifier_gpu0._device == torch.device("cuda:0")
            assert classifier_gpu0._device != torch.device("cuda:1")

        with torch.cuda.device(1):
            classifier_gpu1 = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=loss_fn,
                optimizer=optimizer,
                input_shape=(1, 28, 28),
                nb_classes=10,
            )
            assert classifier_gpu1._device == torch.device("cuda:1")
            assert classifier_gpu1._device != torch.device("cuda:0")


@pytest.mark.only_with_platform("pytorch")
def test_pickle(get_default_mnist_subset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    from art.config import ART_DATA_PATH
    full_path = os.path.join(ART_DATA_PATH, "my_classifier")
    folder = os.path.split(full_path)[0]
    if not os.path.exists(folder):
        os.makedirs(folder)

    model = Model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    myclassifier_2 = PyTorchClassifier(
        model=model, clip_values=(0, 1), loss=loss_fn, optimizer=optimizer, input_shape=(1, 28, 28), nb_classes=10
    )
    myclassifier_2.fit(x_train_mnist, y_train_mnist, batch_size=100, nb_epochs=1)

    pickle.dump(myclassifier_2, open(full_path, "wb"))

    with open(full_path, "rb") as f:
        loaded_model = pickle.load(f)
        np.testing.assert_equal(myclassifier_2._clip_values, loaded_model._clip_values)
        assert myclassifier_2._channel_index == loaded_model._channel_index
        assert set(myclassifier_2.__dict__.keys()) == set(loaded_model.__dict__.keys())

    # Test predict
    predictions_1 = myclassifier_2.predict(x_test_mnist)
    accuracy_1 = np.sum(np.argmax(predictions_1, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
    predictions_2 = loaded_model.predict(x_test_mnist)
    accuracy_2 = np.sum(np.argmax(predictions_2, axis=1) == np.argmax(y_test_mnist, axis=1)) / y_test_mnist.shape[0]
    assert accuracy_1 == accuracy_2


@pytest.mark.only_with_platform("pytorch")
def test_set_learning(get_image_classifier_list):
    classifier, _ = get_image_classifier_list(one_classifier=True)
    assert classifier._model.training
    classifier.set_learning_phase(False)
    assert classifier._model.training is False
    classifier.set_learning_phase(True)
    assert classifier._model.training
    assert classifier.learning_phase


@pytest.mark.only_with_platform("pytorch")
def test_fit_predict(get_image_classifier_list, get_default_mnist_subset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset
    classifier, _ = get_image_classifier_list(one_classifier=True)
    predictions = classifier.predict(x_test_mnist)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_mnist, axis=1)) / x_test_mnist.shape[0]
    logger.info("Accuracy after fitting: %.2f%%", (accuracy * 100))
    assert accuracy == 0.32


@pytest.mark.only_with_platform("pytorch")
def test_fit_image_generator(get_image_classifier_list, image_data_generator, get_default_mnist_subset):
    classifier, _ = get_image_classifier_list(one_classifier=True)

    expected_values = {"pre_fit_accuracy": ExpectedValue(0.32, 0.06), "post_fit_accuracy": ExpectedValue(0.73, 0.06)}

    data_gen = image_data_generator()
    backend_test_fit_generator(expected_values, classifier, data_gen, get_default_mnist_subset, nb_epochs=2)


@pytest.mark.only_with_platform("pytorch")
def test_class_gradient(get_image_classifier_list, get_default_mnist_subset, framework):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_default_mnist_subset

    classifier_logits, _ = get_image_classifier_list(one_classifier=True, from_logits=True)

    # expected_gradients_1_all_labels = np.asarray(
    #     [
    #         -0.00367321,
    #         -0.0002892,
    #         0.00037825,
    #         -0.00053344,
    #         0.00192121,
    #         0.00112047,
    #         0.0023135,
    #         0.0,
    #         0.0,
    #         -0.00391743,
    #         -0.0002264,
    #         0.00238103,
    #         -0.00073711,
    #         0.00270405,
    #         0.00389043,
    #         0.00440818,
    #         -0.00412769,
    #         -0.00441795,
    #         0.00081916,
    #         -0.00091284,
    #         0.00119645,
    #         -0.00849089,
    #         0.00547925,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #     ]
    # )
    #
    # expected_gradients_2_all_labels = np.asarray(
    #     [
    #         -1.0557442e-03,
    #         -1.0079540e-03,
    #         -7.7426381e-04,
    #         1.7387437e-03,
    #         2.1773505e-03,
    #         5.0880131e-05,
    #         1.6497375e-03,
    #         2.6113102e-03,
    #         6.0904315e-03,
    #         4.1080985e-04,
    #         2.5268074e-03,
    #         -3.6661496e-04,
    #         -3.0568994e-03,
    #         -1.1665225e-03,
    #         3.8904310e-03,
    #         3.1726388e-04,
    #         1.3203262e-03,
    #         -1.1720933e-04,
    #         -1.4315107e-03,
    #         -4.7676827e-04,
    #         9.7251305e-04,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #     ]
    # )
    #
    # expected_gradients_1_label5 = np.asarray(
    #     [
    #         -0.00367321,
    #         -0.0002892,
    #         0.00037825,
    #         -0.00053344,
    #         0.00192121,
    #         0.00112047,
    #         0.0023135,
    #         0.0,
    #         0.0,
    #         -0.00391743,
    #         -0.0002264,
    #         0.00238103,
    #         -0.00073711,
    #         0.00270405,
    #         0.00389043,
    #         0.00440818,
    #         -0.00412769,
    #         -0.00441795,
    #         0.00081916,
    #         -0.00091284,
    #         0.00119645,
    #         -0.00849089,
    #         0.00547925,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #     ]
    # )
    #
    # expected_gradients_2_label5 = np.asarray(
    #     [
    #         -1.0557442e-03,
    #         -1.0079540e-03,
    #         -7.7426381e-04,
    #         1.7387437e-03,
    #         2.1773505e-03,
    #         5.0880131e-05,
    #         1.6497375e-03,
    #         2.6113102e-03,
    #         6.0904315e-03,
    #         4.1080985e-04,
    #         2.5268074e-03,
    #         -3.6661496e-04,
    #         -3.0568994e-03,
    #         -1.1665225e-03,
    #         3.8904310e-03,
    #         3.1726388e-04,
    #         1.3203262e-03,
    #         -1.1720933e-04,
    #         -1.4315107e-03,
    #         -4.7676827e-04,
    #         9.7251305e-04,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #     ]
    # )
    #
    # expected_gradients_1_label_array = np.asarray(
    #     [
    #         -0.00195835,
    #         -0.00134457,
    #         -0.00307221,
    #         -0.00340564,
    #         0.00175022,
    #         -0.00239714,
    #         -0.00122619,
    #         0.0,
    #         0.0,
    #         -0.00520899,
    #         -0.00046105,
    #         0.00414874,
    #         -0.00171095,
    #         0.00429184,
    #         0.0075138,
    #         0.00792443,
    #         0.0019566,
    #         0.00035517,
    #         0.00504575,
    #         -0.00037397,
    #         0.00022343,
    #         -0.00530035,
    #         0.0020528,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #     ]
    # )
    #
    # expected_gradients_2_label_array = np.asarray(
    #     [
    #         5.0867130e-03,
    #         4.8564533e-03,
    #         6.1040395e-03,
    #         8.6531248e-03,
    #         -6.0958802e-03,
    #         -1.4114541e-02,
    #         -7.1085966e-04,
    #         -5.0330797e-04,
    #         1.2943064e-02,
    #         8.2416134e-03,
    #         -1.9859453e-04,
    #         -9.8110031e-05,
    #         -3.8902226e-03,
    #         -1.2945874e-03,
    #         7.5138002e-03,
    #         1.7720887e-03,
    #         3.1399354e-04,
    #         2.3657191e-04,
    #         -3.0891625e-03,
    #         -1.0211228e-03,
    #         2.0828887e-03,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #     ]
    # )
    #
    # decimal_precision = 4
    #
    # expected_values = {
    #     "expected_gradients_1_all_labels": ExpectedValue(
    #         expected_gradients_1_all_labels,
    #         decimal_precision,
    #     ),
    #     "expected_gradients_2_all_labels": ExpectedValue(
    #         expected_gradients_2_all_labels,
    #         decimal_precision,
    #     ),
    #     "expected_gradients_1_label5": ExpectedValue(
    #         expected_gradients_1_label5,
    #         decimal_precision,
    #     ),
    #     "expected_gradients_2_label5": ExpectedValue(
    #         expected_gradients_2_label5,
    #         decimal_precision,
    #     ),
    #     "expected_gradients_1_labelArray": ExpectedValue(
    #         expected_gradients_1_label_array,
    #         decimal_precision,
    #     ),
    #     "expected_gradients_2_labelArray": ExpectedValue(
    #         expected_gradients_2_label_array,
    #         decimal_precision,
    #     ),
    # }

    expected_values = {
        "expected_gradients_1_all_labels": ExpectedValue(
            np.asarray(
                [
                    -0.03347399,
                    -0.03195872,
                    -0.02650188,
                    0.04111874,
                    0.08676253,
                    0.03339913,
                    0.06925241,
                    0.09387045,
                    0.15184258,
                    -0.00684002,
                    0.05070481,
                    0.01409407,
                    -0.03632583,
                    0.00151133,
                    0.05102589,
                    0.00766463,
                    -0.00898967,
                    0.00232938,
                    -0.00617045,
                    -0.00201032,
                    0.00410065,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_2_all_labels": ExpectedValue(
            np.asarray(
                [
                    -0.09723657,
                    -0.00240533,
                    0.02445251,
                    -0.00035474,
                    0.04765627,
                    0.04286841,
                    0.07209076,
                    0.0,
                    0.0,
                    -0.07938144,
                    -0.00142567,
                    0.02882954,
                    -0.00049514,
                    0.04170151,
                    0.05102589,
                    0.09544909,
                    -0.04401167,
                    -0.06158172,
                    0.03359772,
                    -0.00838454,
                    0.01722163,
                    -0.13376027,
                    0.08206709,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_1_label5": ExpectedValue(
            np.asarray(
                [
                    -0.03347399,
                    -0.03195872,
                    -0.02650188,
                    0.04111874,
                    0.08676253,
                    0.03339913,
                    0.06925241,
                    0.09387045,
                    0.15184258,
                    -0.00684002,
                    0.05070481,
                    0.01409407,
                    -0.03632583,
                    0.00151133,
                    0.05102589,
                    0.00766463,
                    -0.00898967,
                    0.00232938,
                    -0.00617045,
                    -0.00201032,
                    0.00410065,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_2_label5": ExpectedValue(
            np.asarray(
                [
                    -0.09723657,
                    -0.00240533,
                    0.02445251,
                    -0.00035474,
                    0.04765627,
                    0.04286841,
                    0.07209076,
                    0.0,
                    0.0,
                    -0.07938144,
                    -0.00142567,
                    0.02882954,
                    -0.00049514,
                    0.04170151,
                    0.05102589,
                    0.09544909,
                    -0.04401167,
                    -0.06158172,
                    0.03359772,
                    -0.00838454,
                    0.01722163,
                    -0.13376027,
                    0.08206709,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_1_labelArray": ExpectedValue(
            np.asarray(
                [
                    0.06860766,
                    0.065502,
                    0.08539103,
                    0.13868105,
                    -0.05520725,
                    -0.18788849,
                    0.02264893,
                    0.02980516,
                    0.2226511,
                    0.11288887,
                    -0.00678776,
                    0.02045561,
                    -0.03120914,
                    0.00642691,
                    0.08449504,
                    0.02848018,
                    -0.03251382,
                    0.00854315,
                    -0.02354656,
                    -0.00767687,
                    0.01565931,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
        "expected_gradients_2_labelArray": ExpectedValue(
            np.asarray(
                [
                    -0.0487146,
                    -0.0171556,
                    -0.03161772,
                    -0.0420007,
                    0.03360246,
                    -0.01864819,
                    0.00315916,
                    0.0,
                    0.0,
                    -0.07631349,
                    -0.00374462,
                    0.04229517,
                    -0.01131879,
                    0.05044588,
                    0.08449504,
                    0.12417868,
                    0.07536847,
                    0.03906382,
                    0.09467953,
                    0.00543209,
                    -0.00504872,
                    -0.03366479,
                    -0.00385999,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
            4,
        ),
    }

    labels = np.random.randint(5, size=x_test_mnist.shape[0])
    backend_test_class_gradient(framework, get_default_mnist_subset, classifier_logits, expected_values, labels)


@pytest.mark.only_with_platform("pytorch")
def test_loss_gradient(framework, get_default_mnist_subset, get_image_classifier_list):
    # expected_gradients_1 = np.asarray(
    #     [
    #         7.36792526e-06,
    #         6.50995162e-06,
    #         1.55499711e-05,
    #         1.66183436e-05,
    #         -7.46988326e-06,
    #         1.26695295e-05,
    #         7.61196816e-06,
    #         0.00000000e00,
    #         0.00000000e00,
    #         -1.74639266e-04,
    #         -1.83985649e-05,
    #         1.57154878e-04,
    #         -7.07946092e-05,
    #         1.57594535e-04,
    #         3.20027815e-04,
    #         3.82224127e-04,
    #         2.06750279e-04,
    #         4.05299688e-05,
    #         3.00343090e-04,
    #         5.03358315e-05,
    #         -9.70281690e-07,
    #         -1.66648446e-04,
    #         4.36533046e-05,
    #         0.00000000e00,
    #         0.00000000e00,
    #         0.00000000e00,
    #         0.00000000e00,
    #         0.00000000e00,
    #     ]
    # )
    # expected_gradients_2 = np.asarray(
    #     [
    #         1.6708217e-04,
    #         1.5951888e-04,
    #         1.9378442e-04,
    #         2.3605554e-04,
    #         -1.2112357e-04,
    #         -3.3699317e-04,
    #         5.4395932e-05,
    #         8.7142853e-06,
    #         2.4337447e-04,
    #         9.9849363e-05,
    #         9.5080861e-05,
    #         -7.2551797e-05,
    #         -2.3405801e-04,
    #         -1.4076763e-04,
    #         3.2002782e-04,
    #         1.2220720e-04,
    #         -1.0334983e-04,
    #         3.2093230e-05,
    #         -1.2616906e-04,
    #         -4.1350944e-05,
    #         8.4347754e-05,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #         0.0000000e00,
    #     ]
    # )
    # expected_values = {
    #     "expected_gradients_1": ExpectedValue(
    #         expected_gradients_1,
    #         4,
    #     ),
    #     "expected_gradients_2": ExpectedValue(
    #         expected_gradients_2,
    #         4,
    #     ),
    # }

    expected_values = {
        "expected_gradients_1": ExpectedValue(
            np.asarray(
                [
                    2.10802755e-05,
                    2.13919120e-05,
                    5.20980720e-05,
                    5.48000680e-05,
                    -2.30590031e-05,
                    4.32076595e-05,
                    2.74944887e-05,
                    0.00000000e00,
                    0.00000000e00,
                    -5.83440997e-04,
                    -6.16604229e-05,
                    5.26219024e-04,
                    -2.37398461e-04,
                    5.27310593e-04,
                    1.07115903e-03,
                    1.27738668e-03,
                    6.89289009e-04,
                    1.33779933e-04,
                    1.00320193e-03,
                    1.68109560e-04,
                    -2.86467184e-06,
                    -5.58885862e-04,
                    1.47416518e-04,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ]
            ),
            4,
        ),
        "expected_gradients_2": ExpectedValue(
            np.asarray(
                [
                    5.59206062e-04,
                    5.33892540e-04,
                    6.48919027e-04,
                    7.92516454e-04,
                    -4.02929145e-04,
                    -1.12814642e-03,
                    1.85060024e-04,
                    3.25053406e-05,
                    8.16319487e-04,
                    3.33394884e-04,
                    3.17659928e-04,
                    -2.42046357e-04,
                    -7.81555660e-04,
                    -4.69873514e-04,
                    1.07115903e-03,
                    4.08643362e-04,
                    -3.44107364e-04,
                    1.07128391e-04,
                    -4.22919547e-04,
                    -1.38615724e-04,
                    2.82748661e-04,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ]
            ),
            4,
        ),
    }

    backend_test_loss_gradient(framework, get_default_mnist_subset, get_image_classifier_list, expected_values)
