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

import pytest
from keras_preprocessing.image import ImageDataGenerator
import numpy as np

from art.data_generators import KerasDataGenerator
from art.defences.detector.poison import ActivationDefence
from art.utils import load_mnist
from art.visualization import convert_to_rgb

logger = logging.getLogger(__name__)

NB_TRAIN, NB_TEST, BATCH_SIZE = 300, 10, 128


@pytest.fixture()
def get_ac(get_default_mnist_subset):
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
    x_train, y_train = x_train[:NB_TRAIN], y_train[:NB_TRAIN]
    mnist = (x_train, y_train), (x_test, y_test)
    # mnist = (x_train, y_train), (x_test, y_test), (min_, max_)
    # classifier, _ = image_dl_estimator()
    # Create simple keras model
    import tensorflow as tf
    tf_version = [int(v) for v in tf.__version__.split(".")]
    if tf_version[0] == 2 and tf_version[1] >= 3:
        tf.compat.v1.disable_eager_execution()
        from tensorflow.keras import backend as k
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
    else:
        import keras.backend as k
        from keras.models import Sequential
        from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

    k.set_learning_phase(1)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=x_train.shape[1:]))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Flatten())
    model.add(Dense(10, activation="softmax"))

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    from art.estimators.classification.keras import KerasClassifier

    classifier = KerasClassifier(model=model, clip_values=(min_, max_))

    classifier.fit(x_train, y_train, nb_epochs=1, batch_size=128)

    defence = ActivationDefence(classifier, x_train, y_train)

    datagen = ImageDataGenerator()
    datagen.fit(x_train)

    data_gen = KerasDataGenerator(
        datagen.flow(x_train, y_train, batch_size=NB_TRAIN), size=NB_TRAIN, batch_size=NB_TRAIN
    )

    defence_gen = ActivationDefence(classifier, None, None, generator=data_gen)

    return mnist, classifier, defence, defence_gen


@pytest.mark.parametrize("params", [dict(nb_clusters=0), dict(clustering_method="what"), dict(reduce="what"),
                                    dict(cluster_analysis="what")])
@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_wrong_parameters(get_ac, params):
    _, _, defence, defence_gen = get_ac
    with pytest.raises(ValueError):
        defence.set_params(**params)
    with pytest.raises(ValueError):
        defence_gen.set_params(**params)


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_activations(get_ac):
    mnist, _, defence, _ = get_ac
    (x_train, _), (_, _) = mnist
    activations = defence._get_activations()
    assert len(x_train) == len(activations)


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_output_clusters(get_ac):
    mnist, classifier, defence, _ = get_ac
    (x_train, _), (_, _) = mnist

    n_classes = classifier.nb_classes
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


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_detect_poison(get_ac):
    mnist, _, defence, defence_gen = get_ac
    (x_train, _), (_, _) = mnist

    _, is_clean_lst = defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
    sum_clean1 = sum(is_clean_lst)

    _, is_clean_lst_gen = defence_gen.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
    sum_clean1_gen = sum(is_clean_lst_gen)

    # Check number of items in is_clean
    assert len(x_train) == len(is_clean_lst)
    assert len(x_train) == len(is_clean_lst_gen)

    # Test right number of clusters
    found_clusters = len(np.unique(defence.clusters_by_class[0]))
    found_clusters_gen = len(np.unique(defence_gen.clusters_by_class[0]))
    assert found_clusters, 2
    assert found_clusters_gen, 2

    _, is_clean_lst = defence.detect_poison(nb_clusters=3, nb_dims=10, reduce="PCA", cluster_analysis="distance")
    _, is_clean_lst_gen = defence_gen.detect_poison(
        nb_clusters=3, nb_dims=10, reduce="PCA", cluster_analysis="distance")
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
    assert sum_clean1 != sum_clean2
    assert sum_clean1_gen != sum_clean2_gen

    kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA", "cluster_analysis": "distance"}
    _, is_clean_lst = defence.detect_poison(**kwargs)
    _, is_clean_lst_gen = defence_gen.detect_poison(**kwargs)
    sum_dist = sum(is_clean_lst)
    sum_dist_gen = sum(is_clean_lst_gen)
    kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA", "cluster_analysis": "smaller"}
    _, is_clean_lst = defence.detect_poison(**kwargs)
    _, is_clean_lst_gen = defence_gen.detect_poison(**kwargs)
    sum_size = sum(is_clean_lst)
    sum_size_gen = sum(is_clean_lst_gen)
    assert sum_dist != sum_size
    assert sum_dist_gen != sum_size_gen


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_evaluate_defense(get_ac):
    mnist, _, defence, defence_gen = get_ac
    (x_train, _), (_, _) = mnist

    kwargs = {"nb_clusters": 2, "nb_dims": 10, "reduce": "PCA"}
    _, _ = defence.detect_poison(**kwargs)
    _, _ = defence_gen.detect_poison(**kwargs)
    is_clean = np.zeros(len(x_train))
    defence.evaluate_defence(is_clean)
    defence_gen.evaluate_defence(is_clean)


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_analyze_cluster(get_ac):
    mnist, classifier, defence, defence_gen = get_ac
    (x_train, _), (_, _) = mnist

    defence.analyze_clusters(cluster_analysis="relative-size")
    defence_gen.analyze_clusters(cluster_analysis="relative-size")

    defence.analyze_clusters(cluster_analysis="silhouette-scores")
    defence_gen.analyze_clusters(cluster_analysis="silhouette-scores")

    report, dist_clean_by_class = defence.analyze_clusters(cluster_analysis="distance")
    report_gen, dist_clean_by_class_gen = defence_gen.analyze_clusters(cluster_analysis="distance")
    n_classes = classifier.nb_classes
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
    n_classes = classifier.nb_classes
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
    assert sum_dis != sum_sz, "This is very unlikely to happen... there may be an error"
    assert sum_dis_gen != sum_sz_gen, "This is very unlikely to happen... there may be an error"


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_plot_clusters(get_ac):
    _, _, defence, defence_gen = get_ac
    defence.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
    defence_gen.detect_poison(nb_clusters=2, nb_dims=10, reduce="PCA")
    defence.plot_clusters(save=False)
    defence_gen.plot_clusters(save=False)


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_pickle(get_ac):
    _, classifier, _, _ = get_ac

    # Test pickle and unpickle:
    filename = "test_pickle.h5"
    ActivationDefence._pickle_classifier(classifier, filename)
    loaded = ActivationDefence._unpickle_classifier(filename)

    np.testing.assert_equal(classifier._clip_values, loaded._clip_values)
    assert classifier._channels_first == loaded._channels_first
    assert classifier._use_logits == loaded._use_logits
    assert classifier._input_layer == loaded._input_layer

    ActivationDefence._remove_pickle(filename)


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_fix_relabel_poison(get_ac):
    mnist, classifier, _, _ = get_ac
    (x_train, y_train), (_, _) = mnist
    x_poison = x_train[:100]
    y_fix = y_train[:100]

    test_set_split = 0.7
    n_train = int(len(x_poison) * test_set_split)
    x_test = x_poison[n_train:]
    y_test = y_fix[n_train:]

    predictions = np.argmax(classifier.predict(x_test), axis=1)
    ini_miss = 1 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]

    improvement, new_classifier = ActivationDefence.relabel_poison_ground_truth(
        classifier,
        x_poison,
        y_fix,
        test_set_split=test_set_split,
        tolerable_backdoor=0.01,
        max_epochs=5,
        batch_epochs=10,
    )

    predictions = np.argmax(new_classifier.predict(x_test), axis=1)
    final_miss = 1 - np.sum(predictions == np.argmax(y_test, axis=1)) / y_test.shape[0]

    assert improvement == ini_miss - final_miss

    # Other method (since it's cross validation we can't assert to a concrete number).
    improvement, _ = ActivationDefence.relabel_poison_cross_validation(
        classifier, x_poison, y_fix, n_splits=2, tolerable_backdoor=0.01, max_epochs=5, batch_epochs=10
    )
    assert improvement >= 0


@pytest.mark.skipMlFramework("non_dl_frameworks", "pytorch", "mxnet", "tensorflow")
def test_visualizations(get_ac):
    mnist, _, defence, _ = get_ac
    # test that visualization doesn't error in grayscale and RGB settings
    (x_train, _), (_, _) = mnist
    defence.visualize_clusters(x_train)

    x_train_rgb = convert_to_rgb(x_train)
    defence.visualize_clusters(x_train_rgb)
