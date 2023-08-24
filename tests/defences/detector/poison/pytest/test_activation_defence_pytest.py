# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2018
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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

from keras_preprocessing.image import ImageDataGenerator
import numpy as np
import pytest

from art.data_generators import KerasDataGenerator, PyTorchDataGenerator
from art.defences.detector.poison import ActivationDefence
from art.utils import load_mnist, load_cifar10
from art.visualization import convert_to_rgb
from tests.utils import master_seed, get_image_classifier_hf

logger = logging.getLogger(__name__)

NB_TRAIN, NB_TEST, BATCH_SIZE = 300, 10, 128
HF_MODEL_SIZE = "SMALL"


@pytest.fixture()
def get_data(framework):
    if framework == "huggingface" and HF_MODEL_SIZE == "LARGE":
        import torch

        (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        upsampler = torch.nn.Upsample(scale_factor=7, mode="nearest")
        x_train = np.float32(np.moveaxis(x_train, -1, 1))
        x_test = np.float32(np.moveaxis(x_test, -1, 1))
        x_train = upsampler(torch.from_numpy(x_train)).cpu().numpy()
        x_test = upsampler(torch.from_numpy(x_test)).cpu().numpy()
    elif framework == "huggingface" and HF_MODEL_SIZE == "SMALL":
        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_train = np.float32(np.moveaxis(x_train, -1, 1))
        x_test = np.float32(np.moveaxis(x_test, -1, 1))
    elif framework == "pytorch":
        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
        x_train = np.float32(np.moveaxis(x_train, -1, 1))
        x_test = np.float32(np.moveaxis(x_test, -1, 1))
    else:
        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
        x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]

    return (x_train, y_train), (x_test, y_test), min_, max_


@pytest.fixture()
def get_data_gen(framework, get_data):
    (x_train, y_train), (x_test, y_test), min_, max_ = get_data
    data_gen = None

    if framework == "tensorflow2":
        datagen = ImageDataGenerator()
        datagen.fit(x_train)
        data_gen = KerasDataGenerator(
            datagen.flow(x_train, y_train, batch_size=NB_TRAIN), size=NB_TRAIN, batch_size=NB_TRAIN
        )
    elif framework in ["pytorch", "huggingface"]:
        from torch.utils.data import Dataset

        class CustomImageDataset(Dataset):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

        from torch.utils.data import DataLoader

        train_dataloader = DataLoader(CustomImageDataset(x_train, y_train), batch_size=NB_TRAIN, shuffle=True)
        data_gen = PyTorchDataGenerator(train_dataloader, size=NB_TRAIN, batch_size=NB_TRAIN)

    return data_gen


@pytest.fixture()
def get_mnist_classifier(framework, image_dl_estimator, get_data):
    def _get_classifier():
        classifier = None

        if framework == "huggingface":
            if HF_MODEL_SIZE == "LARGE":
                import torch
                import transformers
                from art.estimators.hugging_face import HuggingFaceClassifier

                model = transformers.AutoModelForImageClassification.from_pretrained(
                    "facebook/deit-tiny-patch16-224",
                    # takes 3 min
                    ignore_mismatched_sizes=True,
                    num_labels=10,
                )

                print("num of parameters is ", sum(p.numel() for p in model.parameters() if p.requires_grad))
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                classifier = HuggingFaceClassifier(
                    model,
                    loss=torch.nn.CrossEntropyLoss(),
                    input_shape=(3, 224, 224),
                    nb_classes=10,
                    optimizer=optimizer,
                    processor=None,
                )
            if HF_MODEL_SIZE == "SMALL":
                classifier = get_image_classifier_hf()
        elif framework == "pytorch":
            import torch

            class Model(torch.nn.Module):
                def __init__(self, number_of_classes: int):
                    super(Model, self).__init__()

                    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                    self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
                    self.relu = torch.nn.ReLU()

                    self.mpool = torch.nn.MaxPool2d(kernel_size=(3, 3))
                    # Problem: Pytorch does not register flatten as a layer.
                    self.fc1 = torch.nn.Linear(in_features=2048, out_features=number_of_classes)

                def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                    """
                    Computes the forward pass though the neural network
                    :param x: input data of shape (batch size, N features)
                    :return: model prediction
                    """
                    x = self.relu(self.conv_1(x))
                    x = self.mpool(x)
                    x = torch.flatten(x, 1)
                    return self.fc1(x)

            model = Model(10)
            from art.estimators.classification import PyTorchClassifier

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

            classifier = PyTorchClassifier(
                model=model,
                clip_values=(0, 1),
                loss=criterion,
                optimizer=optimizer,
                input_shape=(1, 28, 28),
                nb_classes=10,
            )
            # classifier = get_image_classifier_pt()

        elif framework == "tensorflow2":
            # Create simple keras model
            import tensorflow as tf
            from art.estimators.classification.keras import KerasClassifier

            tf_version = [int(v) for v in tf.__version__.split(".")]
            if tf_version[0] == 2 and tf_version[1] >= 3:
                tf.compat.v1.disable_eager_execution()
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
            else:
                from keras.models import Sequential
                from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
            model.add(MaxPooling2D(pool_size=(3, 3)))
            model.add(Flatten())
            model.add(Dense(10, activation="softmax"))

            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

            classifier = KerasClassifier(model=model, clip_values=(min_, max_))

        return classifier

    (x_train, y_train), (x_test, y_test), min_, max_ = get_data
    classifier = _get_classifier()
    classifier.fit(x_train, y_train, nb_epochs=1, batch_size=128)

    return classifier


# Pytorch can be added in but needs a refactor to the core estimator first
@pytest.mark.only_with_platform("tensorflow2", "huggingface")
class TestActivationDefence:
    pytest.classifier = None

    def test_get_classifiers(self, get_mnist_classifier):
        if pytest.classifier is None:
            pytest.classifier = get_mnist_classifier

    def test_layer_counts(self, framework, get_data):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data

        layer_names = pytest.classifier.layer_names
        print("the layer names are ", layer_names)
        for layer_num in range(len(layer_names)):
            activations = pytest.classifier.get_activations(x_train, layer=layer_num, batch_size=64)
            print(f"At layer num {layer_num} the shape is {activations.shape}")

    def setUp(self):
        # Set master seed
        master_seed(1234)

    def test_wrong_parameters_1(self, get_data):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        with pytest.raises(ValueError):
            defence.set_params(nb_clusters=0)

    def test_wrong_parameters_2(self, get_data):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        with pytest.raises(ValueError):
            defence.set_params(clustering_method="what")

    def test_wrong_parameters_3(self, get_data):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        with pytest.raises(ValueError):
            defence.set_params(reduce="what")

    def test_wrong_parameters_4(self, get_data):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        with pytest.raises(ValueError):
            defence.set_params(cluster_analysis="what")

    def test_wrong_parameters_5(self, get_data):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        with pytest.raises(ValueError):
            defence.set_params(ex_re_threshold=-1)

    def test_activations(self, get_data):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        activations = defence._get_activations()
        assert len(x_train) == len(activations)

    def test_output_clusters(self, get_data):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        defence = ActivationDefence(pytest.classifier, x_train, y_train)

        n_classes = pytest.classifier.nb_classes
        for nb_clusters in range(2, 5):
            clusters_by_class, _ = defence.cluster_activations(nb_clusters=nb_clusters)

            # Verify expected number of classes
            assert np.shape(clusters_by_class)[0] == n_classes
            # Check we get the expected number of clusters:
            found_clusters = len(np.unique(clusters_by_class[0]))
            assert found_clusters == nb_clusters
            # Check right amount of data
            n_dp = 0
            for i in range(0, n_classes):
                n_dp += len(clusters_by_class[i])
            assert len(x_train) == n_dp

    def test_detect_poison(self, get_data, get_data_gen):
        data_gen = get_data_gen

        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        defence_gen = ActivationDefence(pytest.classifier, None, None, generator=data_gen)

        _, is_clean_lst = defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA", ex_re_threshold=1)
        sum_clean1 = sum(is_clean_lst)

        _, is_clean_lst_gen = defence_gen.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA", ex_re_threshold=1)
        sum_clean1_gen = sum(is_clean_lst_gen)

        # Check number of items in is_clean
        assert len(x_train) == len(is_clean_lst)
        assert len(x_train) == len(is_clean_lst_gen)

        # Test right number of clusters
        found_clusters = len(np.unique(defence.clusters_by_class[0]))
        found_clusters_gen = len(np.unique(defence_gen.clusters_by_class[0]))
        assert found_clusters == 2
        assert found_clusters_gen == 2

        _, is_clean_lst = defence.detect_poison(
            nb_clusters=3, nb_dims=10, reduce="PCA", cluster_analysis="distance", ex_re_threshold=1
        )
        _, is_clean_lst_gen = defence_gen.detect_poison(
            nb_clusters=3, nb_dims=10, reduce="PCA", cluster_analysis="distance", ex_re_threshold=1
        )
        assert len(x_train) == len(is_clean_lst)
        assert len(x_train) == len(is_clean_lst_gen)

        # Test change of state to new number of clusters:
        found_clusters = len(np.unique(defence.clusters_by_class[0]))
        found_clusters_gen = len(np.unique(defence_gen.clusters_by_class[0]))
        assert found_clusters == 3
        assert found_clusters_gen == 3

        # Test clean data has changed
        sum_clean2 = sum(is_clean_lst)
        sum_clean2_gen = sum(is_clean_lst_gen)
        assert not sum_clean1 == sum_clean2
        assert not sum_clean1_gen == sum_clean2_gen

        kwargs = {
            "nb_clusters": 2,
            "nb_dims": 10,
            "reduce": "PCA",
            "cluster_analysis": "distance",
            "ex_re_threshold": None,
        }
        _, is_clean_lst = defence.detect_poison(**kwargs)
        _, is_clean_lst_gen = defence_gen.detect_poison(**kwargs)
        sum_dist = sum(is_clean_lst)
        sum_dist_gen = sum(is_clean_lst_gen)
        kwargs = {
            "nb_clusters": 2,
            "nb_dims": 10,
            "reduce": "PCA",
            "cluster_analysis": "smaller",
            "ex_re_threshold": None,
        }
        _, is_clean_lst = defence.detect_poison(**kwargs)
        _, is_clean_lst_gen = defence_gen.detect_poison(**kwargs)
        sum_size = sum(is_clean_lst)
        sum_size_gen = sum(is_clean_lst_gen)
        assert not sum_dist == sum_size
        assert not sum_dist_gen == sum_size_gen

    def test_evaluate_defense(self, get_data, get_data_gen):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        data_gen = get_data_gen

        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        defence_gen = ActivationDefence(pytest.classifier, None, None, generator=data_gen)

        kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA"}
        _, _ = defence.detect_poison(**kwargs)
        _, _ = defence_gen.detect_poison(**kwargs)
        is_clean = np.zeros(len(x_train))
        defence.evaluate_defence(is_clean)
        defence_gen.evaluate_defence(is_clean)

    def test_analyze_cluster(self, get_data, get_data_gen):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        data_gen = get_data_gen

        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        defence_gen = ActivationDefence(pytest.classifier, None, None, generator=data_gen)

        defence.analyze_clusters(cluster_analysis="relative-size")
        defence_gen.analyze_clusters(cluster_analysis="relative-size")

        defence.analyze_clusters(cluster_analysis="silhouette-scores")
        defence_gen.analyze_clusters(cluster_analysis="silhouette-scores")

        report, dist_clean_by_class = defence.analyze_clusters(cluster_analysis="distance")
        report_gen, dist_clean_by_class_gen = defence_gen.analyze_clusters(cluster_analysis="distance")
        n_classes = pytest.classifier.nb_classes
        assert n_classes == len(dist_clean_by_class)
        assert n_classes == len(dist_clean_by_class_gen)

        # Check right amount of data
        n_dp = 0
        n_dp_gen = 0
        for i in range(0, n_classes):
            n_dp += len(dist_clean_by_class[i])
            n_dp_gen += len(dist_clean_by_class_gen[i])
        assert len(x_train) == n_dp
        assert len(x_train) == n_dp_gen

        report, sz_clean_by_class = defence.analyze_clusters(cluster_analysis="smaller")
        report_gen, sz_clean_by_class_gen = defence_gen.analyze_clusters(cluster_analysis="smaller")
        n_classes = pytest.classifier.nb_classes
        assert n_classes == len(sz_clean_by_class)
        assert n_classes == len(sz_clean_by_class_gen)

        # Check right amount of data
        n_dp = 0
        n_dp_gen = 0
        sum_sz = 0
        sum_sz_gen = 0
        sum_dis = 0
        sum_dis_gen = 0

        for i in range(0, n_classes):
            n_dp += len(sz_clean_by_class[i])
            n_dp_gen += len(sz_clean_by_class_gen[i])
            sum_sz += sum(sz_clean_by_class[i])
            sum_sz_gen += sum(sz_clean_by_class_gen[i])
            sum_dis += sum(dist_clean_by_class[i])
            sum_dis_gen += sum(dist_clean_by_class_gen[i])
        assert len(x_train) == n_dp
        assert len(x_train) == n_dp_gen

        # Very unlikely that they are the same
        # self.assertNotEqual(sum_dis, sum_sz, msg="This is very unlikely to happen... there may be an error")
        # self.assertNotEqual(sum_dis_gen, sum_sz_gen, msg="This is very unlikely to happen... there may be an error")
        assert not sum_dis == sum_sz
        assert not sum_dis_gen == sum_sz_gen

    def test_plot_clusters(self, get_data, get_data_gen):
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data
        data_gen = get_data_gen

        defence = ActivationDefence(pytest.classifier, x_train, y_train)
        defence_gen = ActivationDefence(pytest.classifier, None, None, generator=data_gen)

        defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
        defence_gen.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
        defence.plot_clusters(save=False)
        defence_gen.plot_clusters(save=False)

    @pytest.mark.only_with_platform("tensorflow2")
    def test_pickle(self):
        # Test pickle and unpickle:
        filename = "test_pickle.h5"
        ActivationDefence._pickle_classifier(pytest.classifier, filename)
        loaded = ActivationDefence._unpickle_classifier(filename)

        np.testing.assert_equal(pytest.classifier._clip_values, loaded._clip_values)
        assert pytest.classifier._channels_first == loaded._channels_first
        assert pytest.classifier._use_logits == loaded._use_logits
        assert pytest.classifier._input_layer == loaded._input_layer

        ActivationDefence._remove_pickle(filename)

    @pytest.mark.only_with_platform("tensorflow2")
    def test_fix_relabel_poison(self, get_data):
        (x_train, y_train), (_, _), min_, max_ = get_data
        x_poison = x_train[:100]
        y_fix = y_train[:100]

        test_set_split = 0.7
        n_train = int(len(x_poison) * test_set_split)
        x_test = x_poison[n_train:]
        y_test = y_fix[n_train:]

        predictions = np.argmax(pytest.classifier.predict(x_test), axis=1)
        ini_miss = 1 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]

        improvement, new_classifier = ActivationDefence.relabel_poison_ground_truth(
            pytest.classifier,
            x_poison,
            y_fix,
            test_set_split=test_set_split,
            tolerable_backdoor=0.01,
            max_epochs=5,
            batch_epochs=10,
        )

        predictions = np.argmax(new_classifier.predict(x_test), axis=1)
        final_miss = 1 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]

        assert improvement == (ini_miss - final_miss)

        # Other method (since it's cross validation we can't assert to a concrete number).
        improvement, _ = ActivationDefence.relabel_poison_cross_validation(
            pytest.classifier, x_poison, y_fix, n_splits=2, tolerable_backdoor=0.01, max_epochs=5, batch_epochs=10
        )
        assert improvement >= 0

    @pytest.mark.only_with_platform("tensorflow2")
    def test_visualizations(self, get_data):
        # test that visualization doesn't error in grayscale and RGB settings
        (x_train, y_train), (x_test, y_test), min_, max_ = get_data

        defence = ActivationDefence(pytest.classifier, x_train, y_train)

        defence.visualize_clusters(x_train)

        x_train_rgb = convert_to_rgb(x_train)
        defence.visualize_clusters(x_train_rgb)
