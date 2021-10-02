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
"""
This module implements the classifier `XGBoostClassifier` for XGBoost models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from copy import deepcopy
import json
import logging
import os
import pickle
from typing import List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np

from art.estimators.classification.classifier import ClassifierDecisionTree
from art.utils import to_categorical
from art import config

if TYPE_CHECKING:
    # pylint: disable=C0412
    import xgboost  # lgtm [py/import-and-import-from]

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
    from art.metrics.verification_decisions_trees import LeafNode, Tree

logger = logging.getLogger(__name__)


class XGBoostClassifier(ClassifierDecisionTree):
    """
    Class for importing XGBoost models.
    """

    estimator_params = ClassifierDecisionTree.estimator_params + [
        "nb_features",
    ]

    def __init__(
        self,
        model: Union["xgboost.Booster", "xgboost.XGBClassifier", None] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        nb_features: Optional[int] = None,
        nb_classes: Optional[int] = None,
    ) -> None:
        """
        Create a `Classifier` instance from a XGBoost model.

        :param model: XGBoost model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param nb_features: The number of features in the training data. Only used if it cannot be extracted from
                             model.
        :param nb_classes: The number of classes in the training data. Only used if it cannot be extracted from model.
        """
        from xgboost import Booster, XGBClassifier

        if not isinstance(model, Booster) and not isinstance(model, XGBClassifier):
            raise TypeError("Model must be of type xgboost.Booster or xgboost.XGBClassifier.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._input_shape = (nb_features,)
        self._nb_classes = self._get_nb_classes(nb_classes)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def nb_features(self) -> int:
        """
        Return the number of features.

        :return: The number of features.
        """
        return self._input_shape[0]  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
                       `fit` function in `xgboost.Booster` or `xgboost.XGBClassifier` and will be passed to this
                       function as such.
        :raises `NotImplementedException`: This method is not supported for XGBoost classifiers.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        import xgboost  # lgtm [py/repeated-import] lgtm [py/import-and-import-from]

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if isinstance(self._model, xgboost.Booster):
            train_data = xgboost.DMatrix(x_preprocessed, label=None)
            y_prediction = self._model.predict(train_data)
            if len(y_prediction.shape) == 1:
                y_prediction = to_categorical(labels=y_prediction, nb_classes=self.nb_classes)
        elif isinstance(self._model, xgboost.XGBClassifier):
            y_prediction = self._model.predict_proba(x_preprocessed)

        # Apply postprocessing
        y_prediction = self._apply_postprocessing(preds=y_prediction, fit=False)

        return y_prediction

    def _get_nb_classes(self, nb_classes: Optional[int]) -> int:
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        """
        from xgboost import Booster, XGBClassifier

        if isinstance(self._model, Booster):
            try:
                return int(len(self._model.get_dump(dump_format="json")) / self._model.n_estimators)  # type: ignore
            except AttributeError:
                if nb_classes is not None:
                    return nb_classes
                raise NotImplementedError(
                    "Number of classes cannot be determined automatically. "
                    + "Please manually set argument nb_classes in XGBoostClassifier."
                ) from AttributeError

        if isinstance(self._model, XGBClassifier):
            return self._model.n_classes_

        return -1

    def save(self, filename: str, path: Optional[str] = None) -> None:
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in
                     the default data location of the library `ART_DATA_PATH`.
        """
        if path is None:
            full_path = os.path.join(config.ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):  # pragma: no cover
            os.makedirs(folder)

        with open(full_path + ".pickle", "wb") as file_pickle:
            pickle.dump(self._model, file=file_pickle)

    def get_trees(self) -> List["Tree"]:
        """
        Get the decision trees.

        :return: A list of decision trees.
        """
        from art.metrics.verification_decisions_trees import Box, Tree

        booster_dump = self._model.get_booster().get_dump(dump_format="json")
        trees = list()

        for i_tree, tree_dump in enumerate(booster_dump):
            box = Box()

            if self._model.n_classes_ == 2:
                class_label = -1
            else:
                class_label = i_tree % self._model.n_classes_

            tree_json = json.loads(tree_dump)
            trees.append(
                Tree(
                    class_id=class_label,
                    leaf_nodes=self._get_leaf_nodes(tree_json, i_tree, class_label, box),
                )
            )

        return trees

    def _get_leaf_nodes(self, node, i_tree, class_label, box) -> List["LeafNode"]:
        from art.metrics.verification_decisions_trees import LeafNode, Box, Interval

        leaf_nodes: List[LeafNode] = list()

        if "children" in node:
            if node["children"][0]["nodeid"] == node["yes"] and node["children"][1]["nodeid"] == node["no"]:
                node_left = node["children"][0]
                node_right = node["children"][1]
            elif node["children"][1]["nodeid"] == node["yes"] and node["children"][0]["nodeid"] == node["no"]:
                node_left = node["children"][1]
                node_right = node["children"][0]
            else:
                raise ValueError

            box_left = deepcopy(box)
            box_right = deepcopy(box)

            feature = int(node["split"][1:])
            box_split_left = Box(intervals={feature: Interval(-np.inf, node["split_condition"])})
            box_split_right = Box(intervals={feature: Interval(node["split_condition"], np.inf)})

            if box.intervals:
                box_left.intersect_with_box(box_split_left)
                box_right.intersect_with_box(box_split_right)
            else:
                box_left = box_split_left
                box_right = box_split_right

            leaf_nodes += self._get_leaf_nodes(node_left, i_tree, class_label, box_left)
            leaf_nodes += self._get_leaf_nodes(node_right, i_tree, class_label, box_right)

        if "leaf" in node:
            leaf_nodes.append(
                LeafNode(
                    tree_id=i_tree,
                    class_label=class_label,
                    node_id=node["nodeid"],
                    box=box,
                    value=node["leaf"],
                )
            )

        return leaf_nodes
