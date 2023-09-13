# MIT License

# Copyright (c) 2023 Yisroel Mirsky

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
This module implements the WHite box membership inference attack
| Paper link: https://arxiv.org/abs/2102.02551

Module author:
Shashank Priyadarshi

Contributed by:
The Offensive AI Research Lab
Ben-Gurion University, Israel
https://offensive-ai-lab.github.io/

Sponsored by INCD

"""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Optional, Union, TYPE_CHECKING

import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.regression import RegressorMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, REGRESSOR_TYPE

logger = logging.getLogger(__name__)


class MembershipInferenceWhiteBox(MembershipInferenceAttack):
    """
    Implementation of a learned white-box membership inference attack.

    This implementation can use as input to the learning process probabilities/logits or losses,
    depending on the type of model and provided configuration.
    """

    attack_params = MembershipInferenceAttack.attack_params + [
        "input_type",
        "attack_model_type",
        "attack_model",
    ]
    _estimator_requirements = (BaseEstimator, (ClassifierMixin, RegressorMixin))

    def __init__(
        self,
        estimator: Union["CLASSIFIER_TYPE", "REGRESSOR_TYPE"],
        input_type: str = "prediction",
        attack_model_type: str = "nn",
        attack_model: Optional[Any] = None,
    ):
        """
        Create a MembershipInferenceWhiteBox attack instance.

        :param estimator: Target estimator.
        :param attack_model_type: the type of default attack model to train, optional. Should be one of `nn` (for neural
                                  network, default), `rf` (for random forest) or `gb` (gradient boosting). If
                                  `attack_model` is supplied, this option will be ignored.
        :param input_type: the type of input to train the attack on. Can be one of: 'prediction' or 'loss'. Default is
                           `prediction`. Predictions can be either probabilities or logits, depending on the return type
                           of the model. If the model is a regressor, only `loss` can be used.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        """

        super().__init__(estimator=estimator)
        self.input_type = input_type
        self.attack_model_type = attack_model_type
        self.attack_model = attack_model

        self._regressor_model = RegressorMixin in type(self.estimator).__mro__

        self._check_params()

        if self.attack_model:
            self.default_model = False
            self.attack_model_type = "None"
        else:
            self.default_model = True
            if self.attack_model_type == "nn":
                import torch
                from torch import nn

                class MembershipInferenceWhiteBoxAttackModel(nn.Module):
                    """
                    Implementation of a pytorch model for learning a membership inference attack.

                    The features used are probabilities/logits or losses for the attack training data along with
                    its true labels.
                    """

                    def __init__(self, num_classes, total, num_features=None):
                        self.num_classes = num_classes
                        if num_features:
                            self.num_features = num_features
                        else:
                            self.num_features = num_classes

                        super().__init__()

                        self.features = nn.Sequential(
                            nn.Dropout(p=0.2),
                            nn.Linear(self.num_classes, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                        )

                        self.loss = nn.Sequential(
                            nn.Dropout(p=0.2),
                            nn.Linear(1, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                        )

                        self.gradient = nn.Sequential(
                            nn.Dropout(p=0.2),
                            nn.Conv2d(1, 1, kernel_size=5, padding=2),
                            nn.BatchNorm2d(1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2),
                            nn.Flatten(),
                            nn.Dropout(p=0.2),
                            nn.Linear(total, 256),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                        )

                        self.labels = nn.Sequential(
                            nn.Dropout(p=0.2),
                            nn.Linear(self.num_classes, 128),
                            nn.ReLU(),
                            nn.Linear(128, 64),
                        )

                        self.combine = nn.Sequential(
                            nn.Dropout(p=0.2),
                            nn.Linear(256, 256),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Linear(64, 1),
                        )

                        self.output = nn.Sigmoid()

                    def forward(self, x_1, loss, gradient, label):
                        """Forward the model."""
                        # x_1 = x_1.view(-1, 28 * 28)
                        out_x1 = self.features(x_1)
                        out_loss = self.loss(loss)
                        out_g = self.gradient(gradient)
                        out_l = self.labels(label)
                        final_inputs = torch.cat((out_x1, out_loss, out_g, out_l), 1)
                        is_member = self.combine(final_inputs)
                        return self.output(is_member)

                if self.input_type == "prediction":
                    num_classes = self.estimator.nb_classes  # type: ignore
                    print(num_classes)
                    gradient_size = self.get_gradient_size()
                    total = gradient_size[0][0] // 2 * gradient_size[0][1] // 2
                    self.attack_model = MembershipInferenceWhiteBoxAttackModel(num_classes, total=total)
                self.epochs = 100
                self.batch_size = 100
                self.learning_rate = 0.0001
            elif self.attack_model_type == "rf":
                self.attack_model = RandomForestClassifier()
            elif self.attack_model_type == "gb":
                self.attack_model = GradientBoostingClassifier()

    def fit(  # pylint: disable=W0613
        self,
        x: np.ndarray,
        y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        pred: Optional[np.ndarray] = None,
        test_pred: Optional[np.ndarray] = None,
        **kwargs
    ):
        """
        Train the attack model.

        :param x: Records that were used in training the target estimator. Can be None if supplying `pred`.
        :param y: True labels for `x`.
        :param test_x: Records that were not used in training the target estimator. Can be None if supplying
                       `test_pred`.
        :param test_y: True labels for `test_x`.
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

        if not self._regressor_model:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=True)
            test_y = check_and_transform_label_format(test_y, nb_classes=self.estimator.nb_classes, return_one_hot=True)

        if x is not None and y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")
        if pred is not None and y.shape[0] != pred.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in pred and y do not match")
        if test_x is not None and test_y.shape[0] != test_x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in test_x and test_y do not match")
        if test_pred is not None and test_y.shape[0] != test_pred.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in test_pred and test_y do not match")

        # Create attack dataset
        # uses final probabilities/logits
        if pred is None:
            x_len = x.shape[0]
        else:
            x_len = pred.shape[0]
        if test_pred is None:
            test_len = test_x.shape[0]
        else:
            test_len = test_pred.shape[0]

        if self.input_type == "prediction":
            # members
            if pred is None:
                if self.attack_model_type == "nn":
                    features = x
                else:
                    features = self.estimator.predict(x).astype(np.float32)
            else:
                features = pred.astype(np.float32)
            # non-members
            if test_pred is None:
                if self.attack_model_type == "nn":
                    test_features = test_x
                else:
                    test_features = self.estimator.predict(test_x).astype(np.float32)
            else:
                test_features = test_pred.astype(np.float32)
        # only for models with loss
        elif self.input_type == "loss":
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
        x_2 = np.concatenate((y, test_y))
        y_new = np.concatenate((labels, test_labels))

        if self._regressor_model:
            x_2 = x_2.astype(np.float32).reshape(-1, 1)

        if self.default_model and self.attack_model_type == "nn":
            import torch
            from torch import nn
            from torch import optim
            from torch.utils.data import DataLoader
            from art.utils import to_cuda

            loss_fn = nn.BCELoss()
            optimizer = optim.Adam(self.attack_model.parameters(), lr=self.learning_rate)  # type: ignore

            attack_train_set = self._get_attack_dataset(f_1=x_1, f_2=x_2, estimator=self.estimator, label=y_new)
            train_loader = DataLoader(attack_train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)

            self.attack_model = to_cuda(self.attack_model)  # type: ignore
            self.attack_model.train()  # type: ignore

            for _ in range(self.epochs):
                for input1, losses, gradients, labels, targets in train_loader:
                    # import ipdb;ipdb.set_trace
                    # targets = targets.type(torch.LongTensor)
                    input1, losses, gradients, labels, targets = (
                        to_cuda(input1),
                        to_cuda(losses),
                        to_cuda(gradients),
                        to_cuda(labels),
                        to_cuda(targets),
                    )
                    _, losses, gradients, labels = (
                        torch.autograd.Variable(input1),
                        torch.autograd.Variable(losses),
                        torch.autograd.Variable(gradients),
                        torch.autograd.Variable(labels),
                    )
                    targets = torch.autograd.Variable(targets)

                    optimizer.zero_grad()

                    outputs = self.attack_model(input1, losses, gradients, labels)  # type: ignore
                    loss = loss_fn(outputs, targets.unsqueeze(1))
                    # import ipdb;ipdb.set_trace()
                    loss.backward()
                    optimizer.step()
        else:
            y_ready = check_and_transform_label_format(y_new, nb_classes=2, return_one_hot=False)
            self.attack_model.fit(np.c_[x_1, x_2], y_ready.ravel())  # type: ignore

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership in the training set of the target estimator.

        :param x: Input records to attack. Can be None if supplying `pred`.
        :param y: True labels for `x`.
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

        if y is None:  # pragma: no cover
            raise ValueError("MembershipInferenceWhiteBox requires true labels `y`.")
        if x is None and pred is None:
            raise ValueError("Must supply either x or pred")

        if self.estimator.input_shape is not None and x is not None:  # pragma: no cover
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of estimator")

        if not self._regressor_model:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=True)

        if y is None:
            raise ValueError("None value detected.")

        if x is not None and y.shape[0] != x.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in x and y do not match")
        if pred is not None and y.shape[0] != pred.shape[0]:  # pragma: no cover
            raise ValueError("Number of rows in pred and y do not match")

        if self.input_type == "prediction":
            if pred is None:
                if self.attack_model_type == "nn":
                    features = x
                else:
                    features = self.estimator.predict(x).astype(np.float32)
            else:
                features = pred.astype(np.float32)
        elif self.input_type == "loss":
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

        if self._regressor_model:
            y = y.astype(np.float32).reshape(-1, 1)

        if self.default_model and self.attack_model_type == "nn":
            import torch
            from torch.utils.data import DataLoader
            from art.utils import to_cuda, from_cuda

            self.attack_model.eval()  # type: ignore
            predictions: Optional[np.ndarray] = None
            test_set = self._get_attack_dataset(f_1=features, f_2=y, estimator=self.estimator)
            test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
            for input1, losses, gradients, labels, targets in test_loader:
                input1, losses, gradients, labels, targets = (
                    to_cuda(input1),
                    to_cuda(losses),
                    to_cuda(gradients),
                    to_cuda(labels),
                    to_cuda(targets),
                )
                outputs = self.attack_model(input1, losses, gradients, labels)  # type: ignore
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
            inferred = self.attack_model.predict(np.c_[features, y])  # type: ignore
            if probabilities:
                inferred_return = inferred
            else:
                inferred_return = np.round(inferred)
        else:
            inferred = self.attack_model.predict_proba(np.c_[features, y])  # type: ignore
            if probabilities:
                inferred_return = inferred[:, [1]]
            else:
                inferred_return = np.round(inferred[:, [1]])

        return inferred_return

    def _get_attack_dataset(self, f_1, f_2, estimator=None, label=None):
        from torch.utils.data.dataset import Dataset

        class AttackDataset(Dataset):
            """
            Implementation of a pytorch dataset for membership inference attack.

            The features are probabilities/logits or losses for the attack training data (`x_1`) along with
            its true labels (`x_2`). The labels (`y`) are a boolean representing whether this is a member.
            """

            def __init__(self, x_1, x_2, y=None):
                import torch
                from torch import nn

                self.x_1 = torch.from_numpy(x_1.astype(np.float64)).type(torch.FloatTensor)
                self.x_2 = torch.from_numpy(x_2.astype(np.int32)).type(torch.FloatTensor)
                self.target_criterion = nn.CrossEntropyLoss(reduction="none")

                if y is not None:
                    self.y = torch.from_numpy(y.astype(np.int8)).type(torch.FloatTensor)
                else:
                    self.y = torch.zeros(x_1.shape[0])

                results = estimator.predict(self.x_1)
                self.losses = self.target_criterion(torch.tensor(results), self.x_2)
                self.losses.requires_grad_(True)
                self.gradients = []
                for loss in self.losses:
                    loss.backward(retain_graph=True)
                    gradient_list = reversed(list(estimator.named_parameters()))
                    for name, parameter in gradient_list:
                        if "weight" in name:
                            gradient = parameter.grad.clone()  # [column[:, None], row].resize_(100,100)
                            gradient = gradient.unsqueeze_(0)
                            self.gradients.append(gradient.unsqueeze_(0))
                            break

                self.labels = []
                class_num = estimator.nb_classes
                tar = np.argmax(self.x_2, axis=1)
                for num in tar:
                    label = [0 for i in range(class_num)]
                    label[num.item()] = 1
                    self.labels.append(label)

                self.gradients = torch.cat(self.gradients, dim=0)
                with torch.no_grad():
                    self.looses = self.losses.unsqueeze_(1).detach()
                self.outputs, _ = torch.sort(torch.tensor(results), descending=True)
                self.labels = torch.Tensor(self.labels)

            def __len__(self):
                return len(self.x_1)

            def __getitem__(self, idx):
                if idx >= len(self.x_1):  # pragma: no cover
                    raise IndexError("Invalid Index")

                return self.outputs[idx], self.losses[idx], self.gradients[idx], self.x_2[idx], self.y[idx]

        return AttackDataset(x_1=f_1, x_2=f_2, y=label)

    def _check_params(self) -> None:
        if self.input_type not in ["prediction", "loss"]:
            raise ValueError("Illegal value for parameter `input_type`.")

        if self._regressor_model:
            if self.input_type != "loss":
                raise ValueError("Illegal value for parameter `input_type` when estimator is a regressor.")

        if self.attack_model_type not in ["nn", "rf", "gb"]:
            raise ValueError("Illegal value for parameter `attack_model_type`.")

        if self.attack_model:
            if ClassifierMixin not in type(self.attack_model).__mro__:
                raise TypeError("Attack model must be of type Classifier.")

    def get_gradient_size(self):
        gradient_size = []
        gradient_list = reversed(list(self.estimator.named_parameters()))
        for name, parameter in gradient_list:
            if "weight" in name:
                gradient_size.append(parameter.shape)

        return gradient_size
