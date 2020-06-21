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

import numpy as np
import numpy.linalg as la

import torch
import torch.utils.data
from torch.autograd import Variable


def get_metrics(model, x_original, x_adv_samples, y):
    model_accuracy_on_non_adversarial_samples, y_pred = evaluate(model, x_original, y)
    model_accuracy_on_adversarial_samples, y_pred_adv = evaluate(model, x_adv_samples, y)

    pert_metric = get_perturbation_metric(x_original, x_adv_samples, y_pred, y_pred_adv, ord=2)
    conf_metric = get_confidence_metric(y_pred, y_pred_adv)

    data = {
        "model accuracy on test data": float(model_accuracy_on_non_adversarial_samples),
        "model accuracy on adversarial samples": float(model_accuracy_on_adversarial_samples),
        "confidence reduced on correctly classified adv_samples": float(conf_metric),
        "average perturbation on misclassified adv_samples": float(pert_metric),
    }
    return data, y_pred, y_pred_adv


# Compute the accuaracy and predicted label using the given test dataset
def evaluate(model, X_test, y_test):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test = torch.utils.data.TensorDataset(
        Variable(torch.FloatTensor(X_test.astype("float32"))), Variable(torch.LongTensor(y_test.astype("float32")))
    )
    test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)
    model.eval()
    correct = 0
    accuracy = 0
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions = torch.softmax(outputs.data, dim=1).detach().numpy()
            correct += predicted.eq(labels.data.view_as(predicted)).sum().item()
            y_pred += predictions.tolist()
        accuracy = 1.0 * correct / len(test_loader.dataset)
    y_pred = np.array(y_pred)
    return accuracy, y_pred


def get_perturbation_metric(x_original, x_adv, y_pred, y_pred_adv, ord=2):
    idxs = np.argmax(y_pred_adv, axis=1) != np.argmax(y_pred, axis=1)

    if np.sum(idxs) == 0.0:
        return 0

    perts_norm = la.norm((x_adv - x_original).reshape(x_original.shape[0], -1), ord, axis=1)
    perts_norm = perts_norm[idxs]

    return np.mean(perts_norm / la.norm(x_original[idxs].reshape(np.sum(idxs), -1), ord, axis=1))


# This computes the change in confidence for all images in the test set
def get_confidence_metric(y_pred, y_pred_adv):
    y_classidx = np.argmax(y_pred, axis=1)
    y_classconf = y_pred[np.arange(y_pred.shape[0]), y_classidx]

    y_adv_classidx = np.argmax(y_pred_adv, axis=1)
    y_adv_classconf = y_pred_adv[np.arange(y_pred_adv.shape[0]), y_adv_classidx]

    idxs = y_classidx == y_adv_classidx

    if np.sum(idxs) == 0.0:
        return 0

    idxnonzero = y_classconf != 0
    idxs = idxs & idxnonzero

    return np.mean((y_classconf[idxs] - y_adv_classconf[idxs]) / y_classconf[idxs])
