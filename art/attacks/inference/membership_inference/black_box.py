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

from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

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
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(
        self,
        classifier: Union["CLASSIFIER_TYPE"],
        input_type: str = "prediction",
        attack_model_type: str = "nn",
        attack_model: Optional[Any] = None,
    ):
        """
        Create a MembershipInferenceBlackBox attack instance.

        :param classifier: Target classifier.
        :param attack_model_type: the type of default attack model to train, optional. Should be one of `nn` (for neural
                                  network, default), `rf` (for random forest) or `gb` (gradient boosting). If
                                  `attack_model` is supplied, this option will be ignored.
        :param input_type: the type of input to train the attack on. Can be one of: 'prediction' or 'loss'. Default is
                           `prediction`. Predictions can be either probabilities or logits, depending on the return type
                           of the model.
        :param attack_model: The attack model to train, optional. If none is provided, a default model will be created.
        """

        super().__init__(estimator=classifier)
        self.input_type = input_type
        self.attack_model_type = attack_model_type
        self.attack_model = attack_model

        self._check_params()

        if self.attack_model:
            self.default_model = False
            self.attack_model_type = "None"
        else:
            self.default_model = True
            if self.attack_model_type == "nn":
                import torch  # lgtm [py/repeated-import]
                import torch.nn as nn  # lgtm [py/repeated-import]

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
                    self.attack_model = MembershipInferenceAttackModel(classifier.nb_classes)
                else:
                    self.attack_model = MembershipInferenceAttackModel(classifier.nb_classes, num_features=1)
                self.epochs = 100
                self.batch_size = 100
                self.learning_rate = 0.0001
            elif self.attack_model_type == "rf":
                self.attack_model = RandomForestClassifier()
            elif self.attack_model_type == "gb":
                self.attack_model = GradientBoostingClassifier()

    def fit(  # pylint: disable=W0613
        self, x: np.ndarray, y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, **kwargs
    ):
        """
        Infer membership in the training set of the target estimator.

        :param x: Records that were used in training the target model.
        :param y: True labels for `x`.
        :param test_x: Records that were not used in training the target model.
        :param test_y: True labels for `test_x`.
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member.
        """
        if self.estimator.input_shape is not None:
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of classifier")
            if self.estimator.input_shape[0] != test_x.shape[1]:
                raise ValueError("Shape of test_x does not match input_shape of classifier")

        y = check_and_transform_label_format(y, len(np.unique(y)), return_one_hot=True)
        test_y = check_and_transform_label_format(test_y, len(np.unique(test_y)), return_one_hot=True)

        if y.shape[0] != x.shape[0]:
            raise ValueError("Number of rows in x and y do not match")
        if test_y.shape[0] != test_x.shape[0]:
            raise ValueError("Number of rows in test_x and test_y do not match")

        # Create attack dataset
        # uses final probabilities/logits
        if self.input_type == "prediction":
            # members
            features = self.estimator.predict(x).astype(np.float32)
            # non-members
            test_features = self.estimator.predict(test_x).astype(np.float32)
        # only for models with loss
        elif self.input_type == "loss":
            if NeuralNetworkMixin not in type(self.estimator).__mro__:
                raise TypeError("loss input_type can only be used with neural networks")
            # members
            features = self.estimator.compute_loss(x, y).astype(np.float32).reshape(-1, 1)
            # non-members
            test_features = self.estimator.compute_loss(test_x, test_y).astype(np.float32).reshape(-1, 1)
        else:
            raise ValueError("Illegal value for parameter `input_type`.")

        # members
        labels = np.ones(x.shape[0])
        # non-members
        test_labels = np.zeros(test_x.shape[0])

        x_1 = np.concatenate((features, test_features))
        x_2 = np.concatenate((y, test_y))
        y_new = np.concatenate((labels, test_labels))

        if self.default_model and self.attack_model_type == "nn":
            import torch  # lgtm [py/repeated-import]
            import torch.nn as nn  # lgtm [py/repeated-import]
            import torch.optim as optim  # lgtm [py/repeated-import]
            from torch.utils.data import DataLoader  # lgtm [py/repeated-import]
            from art.utils import to_cuda

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
                    loss = loss_fn(outputs, targets.unsqueeze(1))  # lgtm [py/call-to-non-callable]

                    loss.backward()
                    optimizer.step()
        else:
            y_ready = check_and_transform_label_format(y_new, len(np.unique(y_new)), return_one_hot=False)
            self.attack_model.fit(np.c_[x_1, x_2], y_ready)  # type: ignore

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Infer membership in the training set of the target estimator.

        :param x: Input records to attack.
        :param y: True labels for `x`.
        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just
                              the predicted class
        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,
                 or class probabilities.
        """
        if y is None:
            raise ValueError("MembershipInferenceBlackBox requires true labels `y`.")

        if self.estimator.input_shape is not None:
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError("Shape of x does not match input_shape of classifier")

        if "probabilities" in kwargs.keys():
            probabilities = kwargs.get("probabilities")
        else:
            probabilities = False

        y = check_and_transform_label_format(y, len(np.unique(y)), return_one_hot=True)

        if y.shape[0] != x.shape[0]:
            raise ValueError("Number of rows in x and y do not match")

        if self.input_type == "prediction":
            features = self.estimator.predict(x).astype(np.float32)
        elif self.input_type == "loss":
            features = self.estimator.compute_loss(x, y).astype(np.float32).reshape(-1, 1)

        if self.default_model and self.attack_model_type == "nn":
            import torch  # lgtm [py/repeated-import]
            from torch.utils.data import DataLoader  # lgtm [py/repeated-import]
            from art.utils import to_cuda, from_cuda

            self.attack_model.eval()  # type: ignore
            inferred: Optional[np.ndarray] = None
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

                if inferred is None:
                    inferred = predicted.detach().numpy()
                else:
                    inferred = np.vstack((inferred, predicted.detach().numpy()))

            if inferred is not None:
                if not probabilities:
                    inferred_return = np.round(inferred)
                else:
                    inferred_return = inferred
            else:
                raise ValueError("No data available.")
        elif not self.default_model:
            # assumes the predict method of the supplied model returns probabilities
            pred = self.attack_model.predict(np.c_[features, y])  # type: ignore
            if probabilities:
                inferred_return = pred
            else:
                inferred_return = np.round(pred)
        else:
            pred = self.attack_model.predict_proba(np.c_[features, y])  # type: ignore
            if probabilities:
                inferred_return = pred[:, [1]]
            else:
                inferred_return = np.round(pred[:, [1]])

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
                import torch  # lgtm [py/repeated-import]

                self.x_1 = torch.from_numpy(x_1.astype(np.float64)).type(torch.FloatTensor)
                self.x_2 = torch.from_numpy(x_2.astype(np.int32)).type(torch.FloatTensor)

                if y is not None:
                    self.y = torch.from_numpy(y.astype(np.int8)).type(torch.FloatTensor)
                else:
                    self.y = torch.zeros(x_1.shape[0])

            def __len__(self):
                return len(self.x_1)

            def __getitem__(self, idx):
                if idx >= len(self.x_1):
                    raise IndexError("Invalid Index")

                return self.x_1[idx], self.x_2[idx], self.y[idx]

        return AttackDataset(x_1=f_1, x_2=f_2, y=label)

    def _check_params(self) -> None:
        if self.input_type not in ["prediction", "loss"]:
            raise ValueError("Illegal value for parameter `input_type`.")

        if self.attack_model_type not in ["nn", "rf", "gb"]:
            raise ValueError("Illegal value for parameter `attack_model_type`.")

        if self.attack_model:
            if ClassifierMixin not in type(self.attack_model).__mro__:
                raise TypeError("Attack model must be of type Classifier.")
