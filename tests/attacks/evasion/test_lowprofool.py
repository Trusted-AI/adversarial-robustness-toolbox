# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
import logging
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from scipy.stats import pearsonr

from art.attacks.evasion import LowProFool
from art.estimators.classification.scikitlearn import ScikitlearnLogisticRegression
from art.estimators.classification import PyTorchClassifier
from art.estimators.classification.scikitlearn import ScikitlearnSVC

logger = logging.getLogger(__name__)


def standardize(data):
    columns = data.columns
    scaler = StandardScaler()
    scaler.fit(data)
    x_scaled = scaler.transform(data)
    return pd.DataFrame(data=x_scaled, columns=columns), scaler


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)


def get_train_and_valid(design_matrix, labels):
    x_train, x_valid, y_train, y_valid = None, None, None, None
    # Split dataset into training and validation sets
    for train_idx, valid_idx in split.split(design_matrix, labels):
        x_train = design_matrix.iloc[train_idx].copy()
        x_valid = design_matrix.iloc[valid_idx].copy()
        y_train = labels.iloc[train_idx].copy()
        y_valid = labels.iloc[valid_idx].copy()

    return x_train, y_train, x_valid, y_valid


def get_irises():
    iris = datasets.load_iris()
    design_matrix_iris = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
    labels_iris = pd.Series(data=iris['target'])

    design_matrix_iris_scaled, scaler_iris = standardize(design_matrix_iris)
    clip_values = (design_matrix_iris_scaled.min(), design_matrix_iris_scaled.max())

    return get_train_and_valid(design_matrix_iris_scaled, labels_iris), clip_values


def get_wines():
    cancer = datasets.load_wine()
    design_matrix_wines = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
    labels_wine = pd.Series(data=cancer['target'])

    design_matrix_wine_scaled, scaler_wine = standardize(design_matrix_wines)
    clip_values = (design_matrix_wine_scaled.min(), design_matrix_wine_scaled.max())

    return get_train_and_valid(design_matrix_wine_scaled, labels_wine), clip_values


def get_breast_cancers():
    cancer = datasets.load_breast_cancer()
    design_matrix_cancer = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
    labels_cancer = pd.Series(data=cancer['target'])

    design_matrix_cancer_scaled, scaler_cancer = standardize(design_matrix_cancer)
    clip_values = (design_matrix_cancer_scaled.min(), design_matrix_cancer_scaled.max())

    return get_train_and_valid(design_matrix_cancer_scaled, labels_cancer), clip_values


class NeuralNetwork:
    def __init__(self):
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

    @staticmethod
    def get_nn_model(input_dimensions, output_dimensions, hidden_neurons):
        return torch.nn.Sequential(
            nn.Linear(input_dimensions, hidden_neurons),
            nn.ReLU(),
            nn.Linear(hidden_neurons, output_dimensions),
            nn.Softmax(dim=1)
        )

    def train_nn(self, nn_model, x, y, learning_rate, epochs):
        optimizer = optim.SGD(nn_model.parameters(), lr=learning_rate)

        for _ in range(epochs):
            y_pred = nn_model.forward(x)
            loss = self.loss_fn(y_pred, y)
            nn_model.zero_grad()
            loss.backward()

            optimizer.step()


class TestLowProFool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (x_train_iris, y_train_iris, x_valid_iris, y_valid_iris), clip_values_iris = get_irises()
        cls.x_train_iris = x_train_iris
        cls.y_train_iris = y_train_iris
        cls.x_valid_iris = x_valid_iris
        cls.y_valid_iris = y_valid_iris
        cls.clip_values_iris = clip_values_iris

        (x_train_wines, y_train_wines, x_valid_wines, y_valid_wines), clip_values_wines = get_wines()
        cls.x_train_wines = x_train_wines
        cls.y_train_wines = y_train_wines
        cls.x_valid_wines = x_valid_wines
        cls.y_valid_wines = y_valid_wines
        cls.clip_values_wines = clip_values_wines

        (x_train_breast_cancer, y_train_breast_cancer, x_valid_breast_cancer,
         y_valid_breast_cancer), clip_values_breast_cancer = get_breast_cancers()
        cls.x_train_breast_cancer = x_train_breast_cancer
        cls.y_train_breast_cancer = y_train_breast_cancer
        cls.x_valid_breast_cancer = x_valid_breast_cancer
        cls.y_valid_breast_cancer = y_valid_breast_cancer
        cls.clip_values_breast_cancer = clip_values_breast_cancer

    def test_general_iris_lr(self):
        """
        Check whether the produced adversaries are correct, given Logistic Regression model and iris flower dataset.
        """
        # setup classifier
        lr_clf = LogisticRegression(penalty='none')
        lr_clf.fit(self.x_train_iris, self.y_train_iris)
        clf_slr = ScikitlearnLogisticRegression(
            model=lr_clf,
            clip_values=self.clip_values_iris
        )

        lpf_slr = LowProFool(classifier=clf_slr, n_steps=25, eta=0.02, lambd=1.5)
        lpf_slr.fit_importances(self.x_train_iris, self.y_train_iris)

        sample = self.x_valid_iris
        # Draw targets different from original labels and then save as one-hot encoded.
        target = np.eye(3)[np.array(
            self.y_valid_iris.apply(lambda x: np.random.choice([i for i in [0, 1, 2] if i != x]))
        )]

        adversaries = lpf_slr.generate(x=sample, y=target)
        expected = np.argmax(target, axis=1)
        predicted = np.argmax(lr_clf.predict_proba(adversaries), axis=1)
        correct = (expected == predicted)
        success_rate = np.sum(correct) / correct.shape[0]
        expected = 0.75
        logger.info("[Irises, Logistic Regression (custom)] success rate of adversarial attack (expected >{:.2f}): "
                    "{:.2f}%".format(expected * 100, success_rate * 100))

        print("===== T E S T =====")
        print("Test name:           {}".format(self.id()))
        print("Classifier:          Logistic Regression (custom)")
        print("Dataset:             Iris flower")
        print("Validation set size: {}".format(target.shape[0]))
        print("Success rate:        {:.2f}%".format(100 * success_rate))
        print()
        self.assertTrue(
            success_rate > expected,
            msg="Failed to achieve {}% accuracy with logistic regression and iris flower dataset".format(expected * 100)
        )

    def test_general_wines_lr(self):
        """
        Check whether the produced adversaries are correct,
        given Logistic Regression classifier and sklearn wines dataset.
        """
        # setup classifier
        lr_clf = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
        lr_clf.fit(self.x_train_wines, self.y_train_wines)
        clf_slr = ScikitlearnLogisticRegression(
            model=lr_clf,
            clip_values=self.clip_values_wines
        )

        lpf_slr = LowProFool(classifier=clf_slr, n_steps=80, eta=0.1, lambd=1.25)
        lpf_slr.fit_importances(self.x_train_wines, self.y_train_wines)

        sample = self.x_valid_wines
        # Draw targets different from original labels and then save as one-hot encoded.
        target = np.eye(3)[np.array(
            self.y_valid_wines.apply(lambda x: np.random.choice([i for i in [0, 1, 2] if i != x]))
        )]

        adversaries = lpf_slr.generate(x=sample, y=target)
        expected = np.argmax(target, axis=1)
        predicted = np.argmax(lr_clf.predict_proba(adversaries), axis=1)
        correct = (expected == predicted)
        success_rate = np.sum(correct) / correct.shape[0]
        expected = 0.75
        logger.info("[Wines, Logistic Regression (custom)] success rate of adversarial attack (expected >{:.2f}):"
                    " {:.2f}%".format(expected * 100, success_rate * 100))

        print("===== T E S T =====")
        print("Test name:           {}".format(self.id()))
        print("Classifier:          Logistic Regression (custom)")
        print("Dataset:             Wines")
        print("Validation set size: {}".format(target.shape[0]))
        print("Success rate:        {:.2f}%".format(100 * success_rate))
        print()
        self.assertTrue(
            success_rate > expected,
            msg="Failed to achieve {}% accuracy with logistic regression and wines dataset".format(expected * 100)
        )

    def test_general_cancer_lr(self):
        """
        Check whether the produced adversaries are correct,
        given Logistic Regression classifier and breast cancer wisconsin dataset.
        """
        # setup classifier
        lr_clf = LogisticRegression(penalty='none', solver='lbfgs', max_iter=100000)
        lr_clf.fit(self.x_train_breast_cancer, self.y_train_breast_cancer)
        clf_slr = ScikitlearnLogisticRegression(
            model=lr_clf,
            clip_values=self.clip_values_breast_cancer
        )

        lpf_slr = LowProFool(classifier=clf_slr, n_steps=30, eta=0.02, lambd=1.5)
        lpf_slr.fit_importances(self.x_train_breast_cancer, self.y_train_breast_cancer)

        sample = self.x_valid_breast_cancer
        # Draw targets different from original labels and then save as one-hot encoded.
        target = np.eye(2)[np.array(
            self.y_valid_breast_cancer.apply(lambda x: np.random.choice([i for i in [0, 1] if i != x]))
        )]

        adversaries = lpf_slr.generate(x=sample, y=target)
        expected = np.argmax(target, axis=1)
        predicted = np.argmax(lr_clf.predict_proba(adversaries), axis=1)
        correct = (expected == predicted)
        success_rate = np.sum(correct) / correct.shape[0]
        expected = 0.75
        logger.info(
            "[Breast cancer, Logistic Regression (custom)] success rate of adversarial attack (expected >{:.2f}): "
            "{:.2f}%".format(expected * 100, success_rate * 100)
        )

        print("===== T E S T =====")
        print("Test name:           {}".format(self.id()))
        print("Classifier:          Logistic Regression (custom)")
        print("Dataset:             Breast cancer")
        print("Validation set size: {}".format(target.shape[0]))
        print("Success rate:        {:.2f}%".format(100 * success_rate))
        print()
        self.assertTrue(
            success_rate > expected,
            msg="Failed to achieve {}% accuracy with logistic regression and breast cancer wisconsin dataset"
                .format(expected * 100)
        )

    def test_general_iris_nn(self):
        """
        Check whether the produced adversaries are correct, given Neural Network classifier and iris flower dataset.
        """
        x = Variable(torch.FloatTensor(np.array(self.x_train_iris)))
        y = Variable(torch.FloatTensor(np.eye(3)[self.y_train_iris]))

        neural_network = NeuralNetwork()
        nn_model_irises = neural_network.get_nn_model(4, 3, 10)
        neural_network.train_nn(nn_model_irises, x, y, 1e-4, 1000)

        est_nn_iris = PyTorchClassifier(model=nn_model_irises,
                                        loss=neural_network.loss_fn, input_shape=(4,), nb_classes=3,
                                        clip_values=self.clip_values_iris)

        lpf_nn = LowProFool(classifier=est_nn_iris, eta=5, lambd=0.2, eta_decay=0.9)

        lpf_nn.fit_importances(self.x_valid_iris, self.y_valid_iris)

        target = np.eye(3)[np.array(
            self.y_valid_iris.apply(lambda x: np.random.choice([i for i in range(3) if i != x]))
        )]

        # Use of LowProFool
        adversaries = lpf_nn.generate(x=self.x_valid_iris, y=target)
        expected = np.argmax(target, axis=1)

        x = Variable(torch.from_numpy(adversaries.astype(np.float32)))
        predicted = np.argmax(nn_model_irises.forward(x).detach().numpy(), axis=1)

        # Test
        correct = (expected == predicted)
        success_rate = np.sum(correct) / correct.shape[0]
        expected = 0.75
        logger.info(
            "[Irises, PyTorch neural network] success rate of adversarial attack (expected >{:.2f}): "
            "{:.2f}%".format(expected * 100, success_rate * 100)
        )

        print("===== T E S T =====")
        print("Test name:           {}".format(self.id()))
        print("Classifier:          PyTorch neural network")
        print("Dataset:             Iris flower")
        print("Validation set size: {}".format(target.shape[0]))
        print("Success rate:        {:.2f}%".format(100 * success_rate))
        print()
        self.assertTrue(
            success_rate > expected,
            msg="Failed to achieve {}% accuracy with neural network and Iris flower dataset".format(expected * 100)
        )

    def test_general_cancer_svc(self):
        """
        Check whether the produced adversaries are correct, given SVC and breast cancer wisconsin dataset.
        """
        svc_clf = SVC()
        svc_clf.fit(self.x_train_breast_cancer, self.y_train_breast_cancer)
        scaled_clip_values_cancer = (-1., 1.)
        clf_svc = ScikitlearnSVC(
            model=svc_clf,
            clip_values=scaled_clip_values_cancer
        )
        lpf_svc = LowProFool(
            classifier=clf_svc,
            n_steps=15,
            eta=15,
            lambd=1.75,
            eta_decay=0.985,
            verbose=False
        )
        lpf_svc.fit_importances(self.x_train_breast_cancer, self.y_train_breast_cancer)
        n_classes = lpf_svc.n_classes
        targets = np.eye(n_classes)[np.array(
            self.y_valid_breast_cancer.apply(lambda x: np.random.choice([i for i in range(n_classes) if i != x]))
        )]
        # Generate adversaries
        adversaries = lpf_svc.generate(x=self.x_valid_breast_cancer, y=targets)

        # Check the success rate
        expected = np.argmax(targets, axis=1)
        predicted = np.argmax(clf_svc.predict(adversaries), axis=1)

        correct = (expected == predicted)
        success_rate = np.sum(correct) / correct.shape[0]
        expected = 0.75
        logger.info(
            "[Breast cancer, Sklearn SVC] success rate of adversarial attack (expected >{:.2f}): "
            "{:.2f}%".format(expected * 100, success_rate * 100)
        )

        print("===== T E S T =====")
        print("Test name:           {}".format(self.id()))
        print("Classifier:          Sklearn SVC")
        print("Dataset:             Breast cancer")
        print("Validation set size: {}".format(targets.shape[0]))
        print("Success rate:        {:.2f}%".format(100 * success_rate))
        print()

        self.assertTrue(
            success_rate > expected,
            msg="Failed to achieve {}% accuracy with SVC and breast cancer wisconsin dataset".format(expected * 100)
        )

    def test_fit_importances(self):
        """
        Check wether feature importance is calculated properly.
        """

        def pearson_correlations(x, y):
            correlations = [pearsonr(x[:, col], y)[0] for col in range(x.shape[1])]
            absolutes = np.abs(np.array(correlations))
            result = absolutes / np.power(np.sum(absolutes ** 2), 0.5)
            return result

        # setup classifier
        lr_clf = LogisticRegression(penalty='none')
        lr_clf.fit(self.x_train_iris, self.y_train_iris)
        clf_slr = ScikitlearnLogisticRegression(
            model=lr_clf,
            clip_values=self.clip_values_iris
        )

        # user defined vector
        vector = pearson_correlations(np.array(self.x_train_iris), np.array(self.y_train_iris))

        # 3 different instances of LowProFool, using 3 different ways of specifiing importance

        # default version - using pearson correlation under the hood
        lpf_slr_default = LowProFool(classifier=clf_slr, n_steps=45, eta=0.02, lambd=1.5, importance='pearson')

        # predefined vector, passed in LowProFool initialization
        lpf_slr_vec = LowProFool(classifier=clf_slr, n_steps=45, eta=0.02, lambd=1.5, importance=vector)

        # user defined function
        lpf_slr_fun = LowProFool(classifier=clf_slr, n_steps=45, eta=0.02, lambd=1.5,
                                 importance=pearson_correlations)

        lpf_slr_default.fit_importances(self.x_train_iris, self.y_train_iris)
        lpf_slr_vec.fit_importances(self.x_train_iris, self.y_train_iris)
        lpf_slr_fun.fit_importances(self.x_train_iris, self.y_train_iris, normalize=False)

        importance_default = lpf_slr_default.importance_vec
        importance_vec_init = lpf_slr_vec.importance_vec
        importance_function = lpf_slr_fun.importance_vec

        # predefined vector passed while fitting
        lpf_slr_default.fit_importances(self.x_train_iris, self.y_train_iris, importance_array=vector)
        importance_vec_fit = lpf_slr_default.importance_vec

        # vector normalization
        vector_norm = vector / np.sum(vector)

        is_default_valid = (vector_norm == importance_default).all()
        is_custom_fun_valid = (vector == importance_function).all()
        is_vec_init_valid = (vector_norm == importance_vec_init).all()
        is_vec_fit_valid = (vector_norm == importance_vec_fit).all()

        logger.info("[Iris flower, Logistic Regression (custom)] Importance fitting test:")
        if not is_default_valid:
            logger.info("Fitting importance by default is invalid")
        elif not is_custom_fun_valid:
            logger.info("Fitting importance with custom function is invalid")
        elif not is_vec_init_valid:
            logger.info("Fitting importance with vector provided in initializer is invalid")
        elif not is_vec_fit_valid:
            logger.info("Fitting importance with vector provided in fit_importances() is invalid")
        else:
            logger.info("Fitting importance with all available methods went successfully")

        print("===== T E S T =====")
        print("Test name:           {}".format(self.id()))
        print("Classifier:          Logistic Regression (custom)")
        print("Dataset:             Iris flower")
        print()

        self.assertTrue(is_default_valid, msg="Fitting importance by default is invalid")
        self.assertTrue(is_custom_fun_valid, msg="Fitting importance with custom function is invalid")
        self.assertTrue(is_vec_init_valid, msg="Fitting importance with provided vector is invalid")
        self.assertTrue(is_vec_fit_valid, msg="Fitting importance with vector provided in fit_importances() is invalid")

    def test_clipping(self):
        """
        Check weather adversaries are clipped properly.
        """
        # setup classifier
        lr_clf = LogisticRegression(penalty='none')
        lr_clf.fit(self.x_train_iris, self.y_train_iris)

        # dataset min-max clipp values
        bottom_min, top_max = self.clip_values_iris
        clf_slr_min_max = ScikitlearnLogisticRegression(
            model=lr_clf,
            clip_values=(bottom_min, top_max)
        )

        # clip values
        bottom_custom = -3
        top_custom = 3
        clf_slr_custom = ScikitlearnLogisticRegression(
            model=lr_clf,
            clip_values=(bottom_custom, top_custom)
        )

        # setting up LowProFool classes with different hyperparameters
        lpf_min_max_default = LowProFool(classifier=clf_slr_min_max, n_steps=45, eta=0.02, lambd=1.5)
        lpf_min_max_high_eta = LowProFool(classifier=clf_slr_min_max, n_steps=45, eta=100000, lambd=1.5)
        lpf_custom_default = LowProFool(classifier=clf_slr_custom, n_steps=45, eta=0.02, lambd=1.5)
        lpf_custom_high_eta = LowProFool(classifier=clf_slr_custom, n_steps=45, eta=100000, lambd=1.5)

        lpf_min_max_default.fit_importances(self.x_train_iris, self.y_train_iris)
        lpf_min_max_high_eta.fit_importances(self.x_train_iris, self.y_train_iris)
        lpf_custom_default.fit_importances(self.x_train_iris, self.y_train_iris)
        lpf_custom_high_eta.fit_importances(self.x_train_iris, self.y_train_iris)

        # generating adversaries
        sample = np.array([[5.5, 2.4, 3.7, 1.0]])
        target = np.array([[0., 0., 1.]])

        adversaries_min_max_default = lpf_min_max_default.generate(x=sample, y=target)
        adversaries_min_max_high_eta = lpf_min_max_high_eta.generate(x=sample, y=target)
        adversaries_custom_default = lpf_custom_default.generate(x=sample, y=target)
        adversaries_custom_high_eta = lpf_custom_high_eta.generate(x=sample, y=target)

        print("===== T E S T =====")
        print("Test name:           {}".format(self.id()))
        print("Classifier:          Logistic Regression (custom)")
        print("Dataset:             Iris flower")
        print()

        # check whether adversaries were clipped properly
        eps = 2e-7
        is_valid_1 = ((bottom_min - eps).to_numpy() <= adversaries_min_max_default).all() and \
                     ((top_max + eps).to_numpy() >= adversaries_min_max_default).all()
        is_valid_2 = ((bottom_min - eps).to_numpy() <= adversaries_min_max_high_eta).all() and \
                     ((top_max + eps).to_numpy() >= adversaries_min_max_high_eta).all()
        is_valid_3 = ((bottom_custom - eps) <= adversaries_custom_default).all() and \
                     ((top_custom + eps) >= adversaries_custom_default).all()
        is_valid_4 = (bottom_custom - eps <= adversaries_custom_high_eta).all() and \
                     (top_custom + eps >= adversaries_custom_high_eta).all()

        is_clipping_valid = is_valid_1 and is_valid_2 and is_valid_3 and is_valid_4
        if is_clipping_valid:
            logger.info("[Iris flower, Logistic Regression (custom)] Clipping is valid.")
        else:
            logger.info("[Iris flower, Logistic Regression (custom)] Clipping is invalid.")

        self.assertTrue(is_valid_1, msg="Bad clipping")
        self.assertTrue(is_valid_2, msg="Bad clipping")
        self.assertTrue(is_valid_3, msg="Bad clipping")
        self.assertTrue(is_valid_4, msg="Bad clipping")


if __name__ == '__main__':
    unittest.main()
