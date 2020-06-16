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
"""Robustness evaluation module."""

import zipfile
import importlib
import re

import numpy as np
from minio import Minio

import torch
import torch.utils.data

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion.fast_gradient import FastGradientMethod

from robustness_util import get_metrics


def robustness_evaluation(
    object_storage_url,
    object_storage_username,
    object_storage_password,
    data_bucket_name,
    result_bucket_name,
    model_id,
    feature_testset_path="processed_data/X_test.npy",
    label_testset_path="processed_data/y_test.npy",
    clip_values=(0, 1),
    nb_classes=2,
    input_shape=(1, 3, 64, 64),
    model_class_file="model.py",
    model_class_name="model",
    LossFn="",
    Optimizer="",
    epsilon=0.2,
):

    url = re.compile(r"https?://")
    cos = Minio(
        url.sub("", object_storage_url),
        access_key=object_storage_username,
        secret_key=object_storage_password,
        secure=False,
    )

    dataset_filenamex = "X_test.npy"
    dataset_filenamey = "y_test.npy"
    weights_filename = "model.pt"
    model_files = model_id + "/_submitted_code/model.zip"

    cos.fget_object(data_bucket_name, feature_testset_path, dataset_filenamex)
    cos.fget_object(data_bucket_name, label_testset_path, dataset_filenamey)
    cos.fget_object(result_bucket_name, model_id + "/" + weights_filename, weights_filename)
    cos.fget_object(result_bucket_name, model_files, "model.zip")

    # Load PyTorch model definition from the source code.
    zip_ref = zipfile.ZipFile("model.zip", "r")
    zip_ref.extractall("model_files")
    zip_ref.close()

    modulename = "model_files." + model_class_file.split(".")[0].replace("-", "_")

    """
    We required users to define where the model class is located or follow
    some naming convention we have provided.
    """
    model_class = getattr(importlib.import_module(modulename), model_class_name)

    # load & compile model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_class().to(device)
    model.load_state_dict(torch.load(weights_filename, map_location=device))

    # Define Loss and optimizer function for the PyTorch model
    if LossFn:
        loss_fn = eval(LossFn)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
    if Optimizer:
        optimizer = eval(Optimizer)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # create pytorch classifier
    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        optimizer=optimizer,
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=clip_values,
    )

    # load test dataset
    x = np.load(dataset_filenamex)
    y = np.load(dataset_filenamey)

    # craft adversarial samples using FGSM
    crafter = FastGradientMethod(classifier, eps=epsilon)
    x_samples = crafter.generate(x)

    # obtain all metrics (robustness score, perturbation metric, reduction in confidence)
    metrics, y_pred_orig, y_pred_adv = get_metrics(model, x, x_samples, y)

    print("metrics:", metrics)
    return metrics
