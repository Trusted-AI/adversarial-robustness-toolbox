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
This module implements the classifier `LightGBMClassifier` for LightGBM models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from copy import deepcopy
import logging
import os
import pickle
from typing import List, Optional, Union, Tuple, TYPE_CHECKING

import numpy as np

from art.estimators.classification.classifier import ClassifierDecisionTree
from art import config

if TYPE_CHECKING:
    # pylint: disable=C0412
    import lightgbm  # lgtm [py/import-and-import-from]

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
    from art.metrics.verification_decisions_trees import LeafNode

logger = logging.getLogger(__name__)


class LightGBMClassifier(ClassifierDecisionTree):
    """
    Class for importing LightGBM models.
    """

    def __init__(
        self,
        model: Optional["lightgbm.Booster"] = None,  # type: ignore
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Create a `Classifier` instance from a LightGBM model.

        :param model: LightGBM model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        """
        from lightgbm import Booster  # type: ignore

        if not isinstance(model, Booster):  # pragma: no cover
            raise TypeError("Model must be of type lightgbm.Booster")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._input_shape = (self._model.num_feature(),)
        self._nb_classes = self._get_nb_classes()

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `lightgbm.Booster` and will be passed to this function as such.
        :raises `NotImplementedException`: This method is not supported for LightGBM classifiers.
        """
        raise NotImplementedError

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.
        """
        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Perform prediction
        predictions = self._model.predict(x_preprocessed)

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=predictions, fit=False)

        return predictions

    def _get_nb_classes(self) -> int:
        """
        Return the number of output classes.

        :return: Number of classes in the data.
        """
        # pylint: disable=W0212
        return self._model._Booster__num_class

    def save(self, filename: str, path: Optional[str] = None) -> None:  # pragma: no cover
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
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(full_path + ".pickle", "wb") as file_pickle:
            pickle.dump(self._model, file=file_pickle)

    def get_trees(self) -> list:
        """
        Get the decision trees.

        :return: A list of decision trees.
        """
        from art.metrics.verification_decisions_trees import Box, Tree

        booster_dump = self._model.dump_model()["tree_info"]
        trees = list()

        for i_tree, tree_dump in enumerate(booster_dump):
            box = Box()

            # pylint: disable=W0212
            if self._model._Booster__num_class == 2:
                class_label = -1
            else:
                class_label = i_tree % self._model._Booster__num_class

            trees.append(
                Tree(
                    class_id=class_label,
                    leaf_nodes=self._get_leaf_nodes(tree_dump["tree_structure"], i_tree, class_label, box),
                )
            )

        return trees

    def _get_leaf_nodes(self, node, i_tree, class_label, box) -> List["LeafNode"]:
        from art.metrics.verification_decisions_trees import Box, Interval, LeafNode

        leaf_nodes: List[LeafNode] = list()

        if "split_index" in node:
            node_left = node["left_child"]
            node_right = node["right_child"]

            box_left = deepcopy(box)
            box_right = deepcopy(box)

            feature = node["split_feature"]
            box_split_left = Box(intervals={feature: Interval(-np.inf, node["threshold"])})
            box_split_right = Box(intervals={feature: Interval(node["threshold"], np.inf)})

            if box.intervals:
                box_left.intersect_with_box(box_split_left)
                box_right.intersect_with_box(box_split_right)
            else:
                box_left = box_split_left
                box_right = box_split_right

            leaf_nodes += self._get_leaf_nodes(node_left, i_tree, class_label, box_left)
            leaf_nodes += self._get_leaf_nodes(node_right, i_tree, class_label, box_right)

        if "leaf_index" in node:
            leaf_nodes.append(
                LeafNode(
                    tree_id=i_tree,
                    class_label=class_label,
                    node_id=node["leaf_index"],
                    box=box,
                    value=node["leaf_value"],
                )
            )

        return leaf_nodes
