# MIT License
#
# Copyright (C) IBM Corporation 2018
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
"""
Module providing convenience functions.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def projection(v, eps, p):
    """
    Project the values in `v` on the L_p norm ball of size `eps`.

    :param v: Array of perturbations to clip.
    :type v: `np.ndarray`
    :param eps: Maximum norm allowed.
    :type eps: `float`
    :param p: L_p norm to use for clipping. Only 1, 2 and `np.Inf` supported for now.
    :type p: `int`
    :return: Values of `v` after projection.
    :rtype: `np.ndarray`
    """
    # Pick a small scalar to avoid division by 0
    tol = 10e-8
    v_ = v.reshape((v.shape[0], -1))

    if p == 2:
        v_ = v_ * np.expand_dims(np.minimum(1., eps / (np.linalg.norm(v_, axis=1) + tol)), axis=1)
    elif p == 1:
        v_ = v_ * np.expand_dims(np.minimum(1., eps / (np.linalg.norm(v_, axis=1, ord=1) + tol)), axis=1)
    elif p == np.inf:
        v_ = np.sign(v_) * np.minimum(abs(v_), eps)
    else:
        raise NotImplementedError('Values of `p` different from 1, 2 and `np.inf` are currently not supported.')

    v = v_.reshape(v.shape)
    return v


def random_sphere(nb_points, nb_dims, radius, norm):
    """
    Generate randomly `m x n`-dimension points with radius `r` and centered around 0.

    :param nb_points: Number of random data points
    :type nb_points: `int`
    :param nb_dims: Dimensionality
    :type nb_dims: `int`
    :param radius: Radius
    :type radius: `float`
    :param norm: Current support: 1, 2, np.inf
    :type norm: `int`
    :return: The generated random sphere
    :rtype: `np.ndarray`
    """
    if norm == 1:
        a = np.zeros(shape=(nb_points, nb_dims + 1))
        a[:, -1] = np.sqrt(np.random.uniform(0, radius ** 2, nb_points))

        for i in range(nb_points):
            a[i, 1:-1] = np.sort(np.random.uniform(0, a[i, -1], nb_dims - 1))

        res = (a[:, 1:] - a[:, :-1]) * np.random.choice([-1, 1], (nb_points, nb_dims))
    elif norm == 2:
        from scipy.special import gammainc

        a = np.random.randn(nb_points, nb_dims)
        s2 = np.sum(a ** 2, axis=1)
        base = gammainc(nb_dims / 2.0, s2 / 2.0) ** (1 / nb_dims) * radius / np.sqrt(s2)
        res = a * (np.tile(base, (nb_dims, 1))).T
    elif norm == np.inf:
        res = np.random.uniform(float(-radius), float(radius), (nb_points, nb_dims))
    else:
        raise NotImplementedError("Norm {} not supported".format(norm))

    return res


def to_categorical(labels, nb_classes=None):
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`
    :type labels: `np.ndarray`
    :param nb_classes: The number of classes (possible labels)
    :type nb_classes: `int`
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`
    :rtype: `np.ndarray`
    """
    labels = np.array(labels, dtype=np.int32)
    if not nb_classes:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical


def random_targets(labels, nb_classes):
    """
    Given a set of correct labels, randomly choose target labels different from the original ones. These can be
    one-hot encoded or integers.

    :param labels: The correct labels
    :type labels: `np.ndarray`
    :param nb_classes: The number of classes for this model
    :type nb_classes: `int`
    :return: An array holding the randomly-selected target classes, one-hot encoded.
    :rtype: `np.ndarray`
    """
    if len(labels.shape) > 1:
        labels = np.argmax(labels, axis=1)

    result = np.zeros(labels.shape)

    for class_ind in range(nb_classes):
        other_classes = list(range(nb_classes))
        other_classes.remove(class_ind)
        in_cl = labels == class_ind
        result[in_cl] = np.random.choice(other_classes)

    return to_categorical(result, nb_classes)


def least_likely_class(x, classifier):
    """
    Compute the least likely class predictions for sample `x`. This strategy for choosing attack targets was used in
    (Kurakin et al., 2016). See https://arxiv.org/abs/1607.02533.

    :param x: A data sample of shape accepted by `classifier`.
    :type x: `np.ndarray`
    :param classifier: The classifier used for computing predictions.
    :type classifier: `Classifier`
    :return: Least-likely class predicted by `classifier` for sample `x` in one-hot encoding.
    :rtype: `np.ndarray`
    """
    return to_categorical(np.argmin(classifier.predict(x), axis=1), nb_classes=classifier.nb_classes)


def get_label_conf(y_vec):
    """
    Returns the confidence and the label of the most probable class given a vector of class confidences
    :param y_vec: (np.ndarray) vector of class confidences, nb of instances as first dimension
    :return: (np.ndarray, np.ndarray) confidences and labels
    """
    assert len(y_vec.shape) == 2

    confs, labels = np.amax(y_vec, axis=1), np.argmax(y_vec, axis=1)
    return confs, labels


def get_labels_np_array(preds):
    """
    Returns the label of the most probable class given a array of class confidences.

    :param preds: (np.ndarray) array of class confidences, nb of instances as first dimension
    :return: (np.ndarray) labels
    """
    preds_max = np.amax(preds, axis=1, keepdims=True)
    y = (preds == preds_max).astype(float)

    return y


def preprocess(x, y, nb_classes=10, max_value=255):
    """Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances
    :type x: `np.ndarray`
    :param y: Labels
    :type y: `np.ndarray`
    :param nb_classes: Number of classes in dataset
    :type nb_classes: `int`
    :param max_value: Original maximum allowed value for features
    :type max_value: `int`
    :return: rescaled values of `x`, `y`
    :rtype: `tuple`
    """
    x = x.astype('float32') / max_value
    y = to_categorical(y, nb_classes)

    return x, y


# -------------------------------------------------------------------------------------------------------- IO FUNCTIONS


def load_cifar10(raw=False):
    """Loads CIFAR10 dataset from config.CIFAR10_PATH or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :type raw: `bool`
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    """
    def load_batch(fpath):
        """
        Utility function for loading CIFAR batches, as written in Keras.

        :param fpath: Full path to the batch file.
        :return: `(data, labels)`
        """
        import sys
        from six.moves import cPickle

        with open(fpath, 'rb') as f:
            if sys.version_info < (3,):
                d = cPickle.load(f)
            else:
                d = cPickle.load(f, encoding='bytes')
                d_decoded = {}
                for k, v in d.items():
                    d_decoded[k.decode('utf8')] = v
                d = d_decoded
        data = d['data']
        labels = d['labels']

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels

    from art import DATA_PATH

    path = get_file('cifar-10-batches-py', extract=True, path=DATA_PATH,
                    url='http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')

    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype=np.uint8)
    y_train = np.zeros((num_train_samples,), dtype=np.uint8)

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
        y_train[(i - 1) * 10000: i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    # Set channels last
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    min_, max_ = 0, 255
    if not raw:
        min_, max_ = 0., 1.
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_mnist(raw=False):
    """Loads MNIST dataset from `DATA_PATH` or downloads it if necessary.

    :param raw: `True` if no preprocessing should be applied to the data. Otherwise, data is normalized to 1.
    :type raw: `bool`
    :return: `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    """
    from art import DATA_PATH

    path = get_file('mnist.npz', path=DATA_PATH, url='https://s3.amazonaws.com/img-datasets/mnist.npz')

    f = np.load(path)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']
    f.close()

    # Add channel axis
    min_, max_ = 0, 255
    if not raw:
        min_, max_ = 0., 1.
        x_train = np.expand_dims(x_train, axis=3)
        x_test = np.expand_dims(x_test, axis=3)
        x_train, y_train = preprocess(x_train, y_train)
        x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_stl():
    """
    Loads the STL-10 dataset from `DATA_PATH` or downloads it if necessary.

    :return: `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    """
    from os.path import join
    from art import DATA_PATH

    min_, max_ = 0., 1.

    # Download and extract data if needed
    path = get_file('stl10_binary', path=DATA_PATH, extract=True,
                    url='https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz')

    with open(join(path, str('train_X.bin')), str('rb')) as f:
        x_train = np.fromfile(f, dtype=np.uint8)
        x_train = np.reshape(x_train, (-1, 3, 96, 96))

    with open(join(path, str('test_X.bin')), str('rb')) as f:
        x_test = np.fromfile(f, dtype=np.uint8)
        x_test = np.reshape(x_test, (-1, 3, 96, 96))

    # Set channel last
    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    with open(join(path, str('train_y.bin')), str('rb')) as f:
        y_train = np.fromfile(f, dtype=np.uint8)
        y_train -= 1

    with open(join(path, str('test_y.bin')), str('rb')) as f:
        y_test = np.fromfile(f, dtype=np.uint8)
        y_test -= 1

    x_train, y_train = preprocess(x_train, y_train)
    x_test, y_test = preprocess(x_test, y_test)

    return (x_train, y_train), (x_test, y_test), min_, max_


def load_dataset(name):
    """
    Loads or downloads the dataset corresponding to `name`. Options are: `mnist`, `cifar10` and `stl10`.

    :param name: Name of the dataset
    :type name: `str`
    :return: The dataset separated in training and test sets as `(x_train, y_train), (x_test, y_test), min, max`
    :rtype: `(np.ndarray, np.ndarray), (np.ndarray, np.ndarray), float, float`
    :raises NotImplementedError: If the dataset is unknown.
    """

    if "mnist" in name:
        return load_mnist()
    elif "cifar10" in name:
        return load_cifar10()
    elif "stl10" in name:
        return load_stl()
    else:
        raise NotImplementedError("There is no loader for dataset '{}'.".format(name))


def _extract(full_path, path):
    import tarfile
    import zipfile
    import shutil

    if full_path.endswith('tar'):
        if tarfile.is_tarfile(full_path):
            archive =  tarfile.open(full_path, "r:")
    elif full_path.endswith('tar.gz'):
        if tarfile.is_tarfile(full_path):
            archive = tarfile.open(full_path, "r:gz")
    elif full_path.endswith('zip'):
        if zipfile.is_zipfile(full_path):
            archive = zipfile.ZipFile(full_path)
        else:
            return False
    else:
        return False

    try:
        archive.extractall(path)
    except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
        if os.path.exists(path):
            if os.path.isfile(path):
                os.remove(path)
            else:
                shutil.rmtree(path)
        raise
    return True


def get_file(filename, url, path=None, extract=False):
    """
    Downloads a file from a URL if it not already in the cache. The file at indicated by `url` is downloaded to the
    path `path` (default is ~/.art/data). and given the name `filename`. Files in tar, tar.gz, tar.bz, and zip formats
    can also be extracted. This is a simplified version of the function with the same name in Keras.

    :param filename: Name of the file.
    :type filename: `str`
    :param url: Download URL.
    :type url: `str`
    :param path: Folder to store the download. If not specified, `~/.art/data` is used instead.
    :type: `str`
    :param extract: If true, tries to extract the archive.
    :type extract: `bool`
    :return: Path to the downloaded file.
    :rtype: `str`
    """
    if path is None:
        from art import DATA_PATH
        path_ = os.path.expanduser(DATA_PATH)
    else:
        path_ = os.path.expanduser(path)
    if not os.access(path_, os.W_OK):
        path_ = os.path.join('/tmp', '.art')
    if not os.path.exists(path_):
        os.makedirs(path_)

    if extract:
        extract_path = os.path.join(path_, filename)
        full_path = extract_path + '.tar.gz'
    else:
        full_path = os.path.join(path_, filename)

    # Determine if dataset needs downloading
    download = not os.path.exists(full_path)

    if download:
        logger.info('Downloading data from %s', url)
        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                from six.moves.urllib.error import HTTPError, URLError
                from six.moves.urllib.request import urlretrieve

                urlretrieve(url, full_path)
            except HTTPError as e:
                raise Exception(error_msg.format(url, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(url, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(full_path):
                os.remove(full_path)
            raise

    if extract:
        if not os.path.exists(extract_path):
            _extract(full_path, path_)
        return extract_path

    return full_path


def make_directory(dir_path):
    """
    Creates the specified tree of directories if needed.

    :param dir_path: (str) directory or file path
    :return: None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_npy_files(path):
    """
    Generator returning all the npy files in path subdirectories.

    :param path: (str) directory path
    :return: (str) paths
    """

    for root, _, files in os.walk(path):
        for file_ in files:
            if file_.endswith(".npy"):
                yield os.path.join(root, file_)


# ------------------------------------------------------------------- ARG PARSER


def get_args(prog, load_classifier=False, load_sample=False, per_batch=False, options=""):
    """
    Parser for all scripts
    :param prog: name of the script calling the function
    :param load_classifier: bool, load a model, default False
    :param load_sample: bool, load (adversarial) data for training, default False
    :param per_batch: bool, load data in batches, default False
    :param options:
    :return: parsed arguments
    """
    parser = argparse.ArgumentParser(prog=prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    option_dict = {
        "a": {"flags": ["-a", "--adv"],
              "kwargs": {"type": str, "dest": 'adv_method', "default": "fgsm",
                         "choices": ["fgsm", "deepfool", "universal", "jsma", "vat", "carlini", "rnd_fgsm"],
                         "help": 'choice of attacker'}},
        "b": {"flags": ["-b", "--batchsize"],
              "kwargs": {"type": int, "dest": 'batch_size', "default": 128, "help": 'size of the batches'}},
        "c": {"flags": ["-c", "--classifier"],
              "kwargs": {"type": str, "dest": 'classifier', "default": "cnn", "choices": ["cnn", "resnet", "mlp"],
                         "help": 'choice of classifier'}},
        "d": {"flags": ["-d", "--dataset"],
              "kwargs": {"type": str, "dest": 'dataset', "default": "mnist",
                         "help": 'either the path or name of the dataset the classifier is tested/trained on.'}},
        "e": {"flags": ["-e", "--epochs"],
              "kwargs": {"type": int, "dest": 'nb_epochs', "default": 20,
                         "help": 'number of epochs for training the classifier'}},
        "f": {"flags": ["-f", "--act"],
              "kwargs": {"type": str, "dest": 'act', "default": "relu", "choices": ["relu", "brelu"],
                         "help": 'choice of activation function'}},
        "n": {"flags": ["-n", "--nbinstances"],
              "kwargs": {"type": int, "dest": 'nb_instances', "default": 1,
                         "help": 'number of supplementary instances per true example'}},
        "r": {"flags": ["-r", "--valsplit"],
              "kwargs": {"type": float, "dest": 'val_split', "default": 0.1,
                         "help": 'ratio of training sample used for validation'}},
        "s": {"flags": ["-s", "--save"],
              "kwargs": {"nargs": '?', "type": str, "dest": 'save', "default": False,
                         "help": 'if set, the classifier is saved; if an argument is provided it is used as path to'
                                 ' store the model'}},
        "t": {"flags": ["-t", "--stdev"],
              "kwargs": {"type": float, "dest": 'std_dev', "default": 0.1,
                         "help": 'standard deviation of the distributions'}},
        "v": {"flags": ["-v", "--verbose"],
              "kwargs": {"dest": 'verbose', "action": "store_true", "help": 'if set, verbose mode'}},
        "z": {"flags": ["-z", "--defences"],
              "kwargs": {"dest": 'defences', "nargs": "*", "default": None, "help": 'list of basic defences.'}},
    }

    # Add required arguments
    if load_classifier:
        parser.add_argument("load", type=str, help='the classifier is loaded from `load` directory.')

    if load_sample:
        parser.add_argument("adv_path", type=str, help='path to the dataset for data augmentation training.')

    if per_batch:
        parser.add_argument("batch_idx", type=int, help='index of the batch to use.')

    # Add optional arguments
    for o in options:
        parser.add_argument(*option_dict[o]["flags"], **option_dict[o]["kwargs"])

    return parser.parse_args()


def get_verbose_print(verbose):
    """
    Sets verbose mode.
    :param verbose: (bool) True for verbose, False for quiet
    :return: (function) printing function
    """
    if verbose:
        return print
    else:
        return lambda *a, **k: None
