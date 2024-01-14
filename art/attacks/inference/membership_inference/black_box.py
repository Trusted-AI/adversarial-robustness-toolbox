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

"""
This module implements membership inference attacks.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Optional, Union, TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.regression import RegressorMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, REGRESSOR_TYPE

logger = logging.getLogger(__name__)


class MembershipInferenceBlackBox(MembershipInferenceAttack):
    """
    Implementation of a learned black-box membership inference attack.

    This implementation can use as input to the learning process probabilities/logits or losses,
    depending on the type of model and provided configuration.
    """

    attack_params = MembershipInferenceAttack.attack_params + [
        "input_type",
        "attack_model_type",
        "attack_model",
        "scaler_type",
        "nn_model_epochs",
        "nn_model_batch_size",
        "nn_model_learning_rate",
    ]
    _estimator_requirements = (BaseEstimator, (ClassifierMixin, RegressorMixin))

    def __init__(
        self,
        estimator: Union["CLASSIFIER_TYPE", "REGRESSOR_TYPE"],
        input_type: str = "prediction",
        attack_model_type: str = "nn",
        attack_model: Optional[Any] = None,
        scaler_type: Optional[str] = "standard",
        nn_model_epochs: int = 100,
        nn_model_batch_size: int = 100,
        nn_model_learning_rate: float = 0.0001,
    ):
        """
        Create a MembershipInferenceBlackBox attack instance.

        :param estimator: Target estimator.
        :param input_type: the type of input to train the attack on. Can be one of: 'prediction' or 'loss'. Default is
                           `prediction`. Predictions can be either probabilities or logits, depending on the return type
                           of the model. If the model is a regressor, only `loss` can be used.
        :param attack_model_type: the type of default attack model to train, optional. Should be one of:
                                 `nn` (neural network, default),
                                 `rf` (random forest),
                                 `gb` (gradient boosting),
                                 `lr` (logistic regression),
                                 `dt` (decision tree),
                                 `knn` (k nearest neighbors),
                                 `svm` (support vector machine).
                                 If `attack_model` is supplied, this option will be ignored.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        :param scaler_type: The type of scaling to apply to the input features to the attack. Can be one of: "standard",
                            "minmax", "robust" or None. If not None, the appropriate scaler from scikit-learn will be
                            applied. If None, no scaling will be applied.
        :param nn_model_epochs: the number of epochs to use when training a nn attack model
        :param nn_model_batch_size: the batch size to use when training a nn attack model
        :param nn_model_learning_rate: the learning rate to use when training a nn attack model
        """

        super().__init__(estimator=estimator)
        self.input_type = input_type
        self.attack_model_type = attack_model_type
        self.attack_model = attack_model
        self.scaler_type = scaler_type
        self.scaler: Optional[Any] = None
        self.epochs = nn_model_epochs
        self.batch_size = nn_model_batch_size
        self.learning_rate = nn_model_learning_rate
        self.use_label = True

        self._regressor_model = RegressorMixin in type(self.estimator).__mro__

        self._check_params()

        if self.attack_model:
            self.default_model = False
            self.attack_model_type = "None"
        else:
            self.default_model = True

            if self.attack_model_type == "rf":
                self.attack_model = RandomForestClassifier()
            elif self.attack_model_type == "gb":
                self.attack_model = GradientBoostingClassifier()
            elif self.attack_model_type == "lr":
                self.attack_model = LogisticRegression()
            elif self.attack_model_type == "dt":
                self.attack_model = DecisionTreeClassifier()
            elif self.attack_model_type == "knn":
                self.attack_model = KNeighborsClassifier()
            elif self.attack_model_type == "svm":
                self.attack_model = SVC(probability=True)
            elif attack_model_type != "nn":
                raise ValueError("Illegal value for parameter `attack_model_type`.")

    def fit(  # pylint: disable=W0613
        self,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        test_x: Optional[np.ndarray] = None,
        test_y: Optional[np.ndarray] = None,
        pred: Optional[np.ndarray] = None,
        test_pred: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Train the attack model.

        :param x: Records that were used in training the target estimator. Can be None if supplying `pred`.
        :param y: True labels for `x`. If not supplied, attack will be based solely on model predictions.
        :param test_x: Records that were not used in training the target estimator. Can be None if supplying
                       `test_pred`.
        :param test_y: True labels for `test_x`. If not supplied, attack will be based solely on model predictions.
        :param pred: Estimator predictions for the records, if not supplied will be generated by calling the estimators'
                     `predict` function. Only relevant for input_type='prediction'.
        :param test_pred: Estimator predictions for the test records, if not supplied will be generated by calling the
                          estimators' `predict` function. Only relevant for input_type='prediction'.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member.
        """
        if x is None and pred is None:
            raise ValueError("Must supply either x or pred")
        if test_x is None and test_pred is None:
            raise ValueError("Must supply either test_x or test_pred")

        if self.estimator.input_shape is not None:
            if x is not None and self.estimator.input_shape[0] != x.shape[1]:  # pragma: no cover
                raise ValueError("Shape of x does not match input_shape of estimator")
            if test_x is not None and self.estimator.input_shape[0] != test_x.shape[1]:  # pragma: no cover
                raise ValueError("Shape of test_x does not match input_shape of estimator")

        if y is not None and test_y is not None and not self._regressor_model:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=True)
            test_y = check_and_transform_label_format(test_y, nb_classes=self.estimator.nb_classes, return_one_hot=True)

        if x is not None and y is not None and y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")
        if pred is not None and y is not None and y.shape[0] != pred.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in pred and y do not match")
        if test_x is not None and test_y is not None and test_y.shape[0] != test_x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in test_x and test_y do not match")
        if test_pred is not None and test_y is not None and test_y.shape[0] != test_pred.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in test_pred and test_y do not match")

        # Create attack dataset
        # uses final probabilities/logits
        x_len = 0
        test_len = 0
        if pred is None and x is not None:
            x_len = x.shape[0]
        elif pred is not None:
            x_len = pred.shape[0]
        if test_pred is None and test_x is not None:
            test_len = test_x.shape[0]
        elif test_pred is not None:
            test_len = test_pred.shape[0]

        if self.input_type == "prediction":
            # members
            if pred is None:
                features = self.estimator.predict(x).astype(np.float32)
            else:
                features = pred.astype(np.float32)
            # non-members
            if test_pred is None:
                test_features = self.estimator.predict(test_x).astype(np.float32)
            else:
                test_features = test_pred.astype(np.float32)
        # only for models with loss
        elif self.input_type == "loss":
            if y is None:
                raise ValueError("Cannot compute loss values without y.")
            if x is not None:
                # members
                features = self.estimator.compute_loss(x, y).astype(np.float32).reshape(-1, 1)
            else:
                try:
                    features = self.estimator.compute_loss_from_predictions(pred, y).astype(np.float32).reshape(-1, 1)
                except NotImplementedError as err:
                    raise ValueError(
                        "For loss input type and no x, the estimator must implement 'compute_loss_from_predictions' "
                        "method"
                    ) from err
            if test_x is not None:
                # non-members
                test_features = self.estimator.compute_loss(test_x, test_y).astype(np.float32).reshape(-1, 1)
            else:
                try:
                    test_features = (
                        self.estimator.compute_loss_from_predictions(test_pred, test_y)
                        .astype(np.float32)
                        .reshape(-1, 1)
                    )
                except NotImplementedError as err:
                    raise ValueError(
                        "For loss input type and no test_x, the estimator must implement "
                        "'compute_loss_from_predictions' method"
                    ) from err
        else:  # pragma: no cover
            raise ValueError("Illegal value for parameter `input_type`.")

        # members
        labels = np.ones(x_len)
        # non-members
        test_labels = np.zeros(test_len)

        x_1 = np.concatenate((features, test_features))
        x_2: Optional[np.ndarray] = None
        if y is not None and test_y is not None:
            x_2 = np.concatenate((y, test_y))
            if self._regressor_model and x_2 is not None:
                x_2 = x_2.astype(np.float32).reshape(-1, 1)
        y_new = np.concatenate((labels, test_labels))
        if x_2 is None:
            self.use_label = False

        if self.scaler_type:
            if self.scaler_type == "standard":
                self.scaler = StandardScaler()
            elif self.scaler_type == "minmax":
                self.scaler = MinMaxScaler()
            elif self.scaler_type == "robust":
                self.scaler = RobustScaler()
            else:
                raise ValueError("Illegal scaler_type: ", self.scaler_type)

        if self.default_model and self.attack_model_type == "nn":
            import torch
            from torch import nn
            from torch import optim
            from torch.utils.data import DataLoader
            from art.utils import to_cuda

            if self.scaler:
                self.scaler.fit(x_1)
                x_1 = self.scaler.transform(x_1)

            if x_2 is not None:

                class MembershipInferenceAttackModel(nn.Module):
                    """
                    Implementation of a pytorch model for learning a membership inference attack.

                    The features used are probabilities/logits or losses for the attack training data along with
                    its true labels.
                    """

                    def __init__(self, num_classes, num_features=None):

                        self.num_classes = num_classes
                        if num_features:
                            self.num_features = num_features
                        else:
                            self.num_features = num_classes

                        super().__init__()

                        self.features = nn.Sequential(
                            nn.Linear(self.num_features, 512),
                            nn.ReLU(),
                            nn.Linear(512, 100),
                            nn.ReLU(),
                            nn.Linear(100, 64),
                            nn.ReLU(),
                        )

                        self.labels = nn.Sequential(
                            nn.Linear(self.num_classes, 256),
                            nn.ReLU(),
                            nn.Linear(256, 64),
                            nn.ReLU(),
                        )

                        self.combine = nn.Sequential(
                            nn.Linear(64 * 2, 1),
                        )

                        self.output = nn.Sigmoid()

                    def forward(self, x_1, label):
                        """Forward the model."""
                        out_x1 = self.features(x_1)
                        out_l = self.labels(label)
                        is_member = self.combine(torch.cat((out_x1, out_l), 1))
                        return self.output(is_member)

                if self.input_type == "prediction":
                    num_classes = self.estimator.nb_classes  # type: ignore
                    self.attack_model = MembershipInferenceAttackModel(num_classes)
                else:  # loss
                    if self._regressor_model:
                        self.attack_model = MembershipInferenceAttackModel(1, num_features=1)
                    else:
                        num_classes = self.estimator.nb_classes  # type: ignore
                        self.attack_model = MembershipInferenceAttackModel(num_classes, num_features=1)

                loss_fn = nn.BCELoss()
                optimizer = optim.Adam(self.attack_model.parameters(), lr=self.learning_rate)  # type: ignore

                attack_train_set = self._get_attack_dataset(f_1=x_1, f_2=x_2, label=y_new)
                train_loader = DataLoader(attack_train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

                self.attack_model = to_cuda(self.attack_model)  # type: ignore
                self.attack_model.train()  # type: ignore

                for _ in range(self.epochs):
                    for (input1, input2, targets) in train_loader:
                        input1, input2, targets = to_cuda(input1), to_cuda(input2), to_cuda(targets)
                        _, input2 = torch.autograd.Variable(input1), torch.autograd.Variable(input2)
                        targets = torch.autograd.Variable(targets)

                        optimizer.zero_grad()
                        outputs = self.attack_model(input1, input2)  # type: ignore
                        loss = loss_fn(outputs, targets.unsqueeze(1))

                        loss.backward()
                        optimizer.step()
            else:  # no label

                class MembershipInferenceAttackModelNoLabel(nn.Module):
                    """
                    Implementation of a pytorch model for learning a membership inference attack.

                    The features used are probabilities/logits or losses for the attack training data along with
                    its true labels.
                    """

                    def __init__(self, num_features):

                        self.num_features = num_features

                        super().__init__()

                        self.features = nn.Sequential(
                            nn.Linear(self.num_features, 512),
                            nn.ReLU(),
                            nn.Linear(512, 100),
                            nn.ReLU(),
                            nn.Linear(100, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1),
                        )

                        self.output = nn.Sigmoid()

                    def forward(self, x_1):
                        """Forward the model."""
                        out_x1 = self.features(x_1)
                        return self.output(out_x1)

                num_classes = self.estimator.nb_classes  # type: ignore
                self.attack_model = MembershipInferenceAttackModelNoLabel(num_classes)

                loss_fn = nn.BCELoss()
                optimizer = optim.Adam(self.attack_model.parameters(), lr=self.learning_rate)  # type: ignore

                attack_train_set = self._get_attack_dataset_no_label(f_1=x_1, label=y_new)
                train_loader = DataLoader(attack_train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

                self.attack_model = to_cuda(self.attack_model)  # type: ignore
                self.attack_model.train()  # type: ignore

                for _ in range(self.epochs):
                    for (input1, targets) in train_loader:
                        input1, targets = to_cuda(input1), to_cuda(targets)
                        input1 = torch.autograd.Variable(input1)
                        targets = torch.autograd.Variable(targets)

                        optimizer.zero_grad()
                        outputs = self.attack_model(input1)  # type: ignore
                        loss = loss_fn(outputs, targets.unsqueeze(1))

                        loss.backward()
                        optimizer.step()

        else:  # not nn
            y_ready = check_and_transform_label_format(y_new, nb_classes=2, return_one_hot=False)
            if x_2 is not None:
                x = np.c_[x_1, x_2]
                if self.scaler:
                    self.scaler.fit(x)
                    x = self.scaler.transform(x)
                self.attack_model.fit(x, y_ready.ravel())  # type: ignore
            else:
                if self.scaler:
                    self.scaler.fit(x_1)
                    x_1 = self.scaler.transform(x_1)
                self.attack_model.fit(x_1, y_ready.ravel())  # type: ignore

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership in the training set of the target estimator.

        :param x: Input records to attack. Can be None if supplying `pred`.
        :param y: True labels for `x`. If not supplied, attack will be based solely on model predictions.
        :param pred: Estimator predictions for the records, if not supplied will be generated by calling the estimators'
                     `predict` function. Only relevant for input_type='prediction'.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                 or class probabilities.
        """
        if "pred" in kwargs:
            pred = kwargs.get("pred")
        else:
            pred = None

        if "probabilities" in kwargs:
            probabilities = kwargs.get("probabilities")
        else:
            probabilities = False

        if x is None and pred is None:
            raise ValueError("Must supply either x or pred")

        if y is None and self.use_label:
            raise ValueError("y must be provided")

        if self.estimator.input_shape is not None and x is not None:  # pragma: no cover
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of estimator")

        if y is not None and not self._regressor_model:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=True)

        if x is not None and y is not None and y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")
        if pred is not None and y is not None and y.shape[0] != pred.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in pred and y do not match")

        if self.input_type == "prediction":
            if pred is None:
                features = self.estimator.predict(x).astype(np.float32)
            else:
                features = pred.astype(np.float32)
        elif self.input_type == "loss":
            if y is None:
                raise ValueError("Cannot compute loss values without y.")
            if x is not None:
                features = self.estimator.compute_loss(x, y).astype(np.float32).reshape(-1, 1)
            else:
                try:
                    features = self.estimator.compute_loss_from_predictions(pred, y).astype(np.float32).reshape(-1, 1)
                except NotImplementedError as err:
                    raise ValueError(
                        "For loss input type and no x, the estimator must implement 'compute_loss_from_predictions' "
                        "method"
                    ) from err
        else:
            raise ValueError("Value of `input_type` not recognized.")

        if y is not None and self._regressor_model:
            y = y.astype(np.float32).reshape(-1, 1)

        if self.default_model and self.attack_model_type == "nn":
            import torch
            from torch.utils.data import DataLoader
            from art.utils import to_cuda, from_cuda

            if self.scaler:
                features = self.scaler.transform(features)

            self.attack_model.eval()  # type: ignore
            predictions: Optional[np.ndarray] = None

            if y is not None and self.use_label:
                test_set = self._get_attack_dataset(f_1=features, f_2=y)
                test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
                for input1, input2, _ in test_loader:
                    input1, input2 = to_cuda(input1), to_cuda(input2)
                    outputs = self.attack_model(input1, input2)  # type: ignore
                    if not probabilities:
                        predicted = torch.round(outputs)
                    else:
                        predicted = outputs
                    predicted = from_cuda(predicted)

                    if predictions is None:
                        predictions = predicted.detach().numpy()
                    else:
                        predictions = np.vstack((predictions, predicted.detach().numpy()))
            else:
                test_set = self._get_attack_dataset_no_label(f_1=features)
                test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
                for input1, _ in test_loader:
                    input1 = to_cuda(input1)
                    outputs = self.attack_model(input1)  # type: ignore
                    if not probabilities:
                        predicted = torch.round(outputs)
                    else:
                        predicted = outputs
                    predicted = from_cuda(predicted)

                    if predictions is None:
                        predictions = predicted.detach().numpy()
                    else:
                        predictions = np.vstack((predictions, predicted.detach().numpy()))
            if predictions is not None:
                if not probabilities:
                    inferred_return = np.round(predictions)
                else:
                    inferred_return = predictions
            else:  # pragma: no cover
                raise ValueError("No data available.")
        elif not self.default_model:
            # assumes the predict method of the supplied model returns probabilities
            if y is not None and self.use_label:
                features = np.c_[features, y]
                if self.scaler:
                    features = self.scaler.transform(features)
                inferred = self.attack_model.predict(features)  # type: ignore
            else:
                if self.scaler:
                    features = self.scaler.transform(features)
                inferred = self.attack_model.predict(features)  # type: ignore
            if probabilities:
                inferred_return = inferred
            else:
                inferred_return = np.round(inferred)
        else:
            if y is not None and self.use_label:
                features = np.c_[features, y]
                if self.scaler:
                    features = self.scaler.transform(features)
                inferred = self.attack_model.predict_proba(features)  # type: ignore
            else:
                if self.scaler:
                    features = self.scaler.transform(features)
                inferred = self.attack_model.predict_proba(features)  # type: ignore
            if probabilities:
                inferred_return = inferred[:, [1]]
            else:
                inferred_return = np.round(inferred[:, [1]])

        return inferred_return

    def _get_attack_dataset(self, f_1, f_2, label=None):
        from torch.utils.data.dataset import Dataset

        class AttackDataset(Dataset):
            """
            Implementation of a pytorch dataset for membership inference attack.

            The features are probabilities/logits or losses for the attack training data (`x_1`) along with
            its true labels (`x_2`). The labels (`y`) are a boolean representing whether this is a member.
            """

            def __init__(self, x_1, x_2, y=None):
                import torch

                self.x_1 = torch.from_numpy(x_1.astype(np.float64)).type(torch.FloatTensor)
                self.x_2 = torch.from_numpy(x_2.astype(np.int32)).type(torch.FloatTensor)

                if y is not None:
                    self.y = torch.from_numpy(y.astype(np.int8)).type(torch.FloatTensor)
                else:
                    self.y = torch.zeros(x_1.shape[0])

            def __len__(self):
                return len(self.x_1)

            def __getitem__(self, idx):
                if idx >= len(self.x_1):  # pragma: no cover
                    raise IndexError("Invalid Index")

                return self.x_1[idx], self.x_2[idx], self.y[idx]

        return AttackDataset(x_1=f_1, x_2=f_2, y=label)

    def _get_attack_dataset_no_label(self, f_1, label=None):
        from torch.utils.data.dataset import Dataset

        class AttackDataset(Dataset):
            """
            Implementation of a pytorch dataset for membership inference attack.

            The features are probabilities/logits or losses for the attack training data (`x_1`) along with
            its true labels (`x_2`). The labels (`y`) are a boolean representing whether this is a member.
            """

            def __init__(self, x_1, y=None):
                import torch

                self.x_1 = torch.from_numpy(x_1.astype(np.float64)).type(torch.FloatTensor)

                if y is not None:
                    self.y = torch.from_numpy(y.astype(np.int8)).type(torch.FloatTensor)
                else:
                    self.y = torch.zeros(x_1.shape[0])

            def __len__(self):
                return len(self.x_1)

            def __getitem__(self, idx):
                if idx >= len(self.x_1):  # pragma: no cover
                    raise IndexError("Invalid Index")

                return self.x_1[idx], self.y[idx]

        return AttackDataset(x_1=f_1, y=label)

    def _check_params(self) -> None:
        if self.input_type not in ["prediction", "loss"]:
            raise ValueError("Illegal value for parameter `input_type`.")

        if self._regressor_model:
            if self.input_type != "loss":
                raise ValueError("Illegal value for parameter `input_type` when estimator is a regressor.")

        if self.attack_model_type not in ["nn", "rf", "gb", "lr", "dt", "knn", "svm"]:
            raise ValueError("Illegal value for parameter `attack_model_type`.")

        if self.attack_model:
            if ClassifierMixin not in type(self.attack_model).__mro__:
                raise TypeError("Attack model must be of type Classifier.")
