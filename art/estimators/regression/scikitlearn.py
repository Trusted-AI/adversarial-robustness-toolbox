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
"""
This module implements the regressors for scikit-learn models.
"""
import logging
import os
import pickle
from copy import deepcopy
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.scikitlearn import ScikitlearnEstimator
from art.estimators.regression.regressor import RegressorMixin
from art import config

if TYPE_CHECKING:
    # pylint: disable=C0412
    import sklearn

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
    from art.metrics.verification_decisions_trees import LeafNode

logger = logging.getLogger(__name__)


class ScikitlearnRegressor(RegressorMixin, ScikitlearnEstimator):  # lgtm [py/missing-call-to-init]
    """
    Wrapper class for scikit-learn regression models.
    """

    estimator_params = ScikitlearnEstimator.estimator_params

    def __init__(
        self,
        model: "sklearn.base.BaseEstimator",
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Create a `Regressor` instance from a scikit-learn regressor model.

        :param model: scikit-learn regressor model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        """
        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._input_shape = self._get_input_shape(model)

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Fit the regressor on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values.
        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the
               `fit` function in `sklearn` regressor and will be passed to this function as such.
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)
        y_preprocessed = np.argmax(y_preprocessed, axis=1)

        self.model.fit(x_preprocessed, y_preprocessed, **kwargs)
        self._input_shape = self._get_input_shape(self.model)

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :return: Array of predictions.
        :raises `ValueError`: If the regressor does not have the method `predict`
        """
        # Apply defences
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if callable(getattr(self.model, "predict", None)):
            y_pred = self.model.predict(x_preprocessed)
        else:
            raise ValueError("The provided model does not have the method `predict`.")

        # Apply postprocessing
        predictions = self._apply_postprocessing(preds=y_pred, fit=False)

        return predictions

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
        if not os.path.exists(folder):
            os.makedirs(folder)

        with open(full_path + ".pickle", "wb") as file_pickle:
            pickle.dump(self.model, file=file_pickle)

    def clone_for_refitting(self) -> "ScikitlearnRegressor":  # lgtm [py/inheritance/incorrect-overridden-signature]
        """
        Create a copy of the classifier that can be refit from scratch.

        :return: new estimator
        """
        import sklearn  # lgtm [py/repeated-import]

        clone = type(self)(sklearn.base.clone(self.model))
        params = self.get_params()
        del params["model"]
        clone.set_params(**params)
        return clone

    def reset(self) -> None:
        """
        Resets the weights of the classifier so that it can be refit from scratch.

        """
        # No need to do anything since scikitlearn models start from scratch each time fit() is called
        pass

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the MSE loss of the regressor for samples `x`.

        :param x: Input samples.
        :param y: Target values.
        :return: Loss values.
        """

        return (y - self.predict(x)) ** 2


class ScikitlearnDecisionTreeRegressor(ScikitlearnRegressor):
    """
    Wrapper class for scikit-learn Decision Tree Regressor models.
    """

    def __init__(
        self,
        model: "sklearn.tree.DecisionTreeRegressor",
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
    ) -> None:
        """
        Create a `Regressor` instance from a scikit-learn Decision Tree Regressor model.

        :param model: scikit-learn Decision Tree Regressor model.
        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
               for features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        """
        # pylint: disable=E0001
        import sklearn  # lgtm [py/repeated-import]

        if not isinstance(model, sklearn.tree.DecisionTreeRegressor):
            raise TypeError("Model must be of type sklearn.tree.DecisionTreeRegressor.")

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self._model = model

    def get_values_at_node(self, node_id: int) -> np.ndarray:
        """
        Returns the feature of given id for a node.

        :return: Normalized values at node node_id.
        """
        return self.model.tree_.value[node_id]

    def get_left_child(self, node_id: int) -> int:
        """
        Returns the id of the left child node of node_id.

        :return: The indices of the left child in the tree.
        """
        return self.model.tree_.children_left[node_id]

    def get_right_child(self, node_id: int) -> int:
        """
        Returns the id of the right child node of node_id.

        :return: The indices of the right child in the tree.
        """
        return self.model.tree_.children_right[node_id]

    def get_decision_path(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the path through nodes in the tree when classifying x. Last one is leaf, first one root node.

        :return: The indices of the nodes in the array structure of the tree.
        """
        if len(np.shape(x)) == 1:
            return self.model.decision_path(x.reshape(1, -1)).indices

        return self.model.decision_path(x).indices

    def get_threshold_at_node(self, node_id: int) -> float:
        """
        Returns the threshold of given id for a node.

        :return: Threshold value of feature split in this node.
        """
        return self.model.tree_.threshold[node_id]

    def get_feature_at_node(self, node_id: int) -> int:
        """
        Returns the feature of given id for a node.

        :return: Feature index of feature split in this node.
        """
        return self.model.tree_.feature[node_id]

    def get_samples_at_node(self, node_id: int) -> int:
        """
        Returns the number of training samples mapped to a node.

        :return: Number of samples mapped this node.
        """
        return self.model.tree_.n_node_samples[node_id]

    def _get_leaf_nodes(self, node_id, i_tree, class_label, box) -> List["LeafNode"]:
        from art.metrics.verification_decisions_trees import LeafNode, Box, Interval

        leaf_nodes: List[LeafNode] = list()

        if self.get_left_child(node_id) != self.get_right_child(node_id):

            node_left = self.get_left_child(node_id)
            node_right = self.get_right_child(node_id)

            box_left = deepcopy(box)
            box_right = deepcopy(box)

            feature = self.get_feature_at_node(node_id)
            box_split_left = Box(intervals={feature: Interval(-np.inf, self.get_threshold_at_node(node_id))})
            box_split_right = Box(intervals={feature: Interval(self.get_threshold_at_node(node_id), np.inf)})

            if box.intervals:
                box_left.intersect_with_box(box_split_left)
                box_right.intersect_with_box(box_split_right)
            else:
                box_left = box_split_left
                box_right = box_split_right

            leaf_nodes += self._get_leaf_nodes(node_left, i_tree, class_label, box_left)
            leaf_nodes += self._get_leaf_nodes(node_right, i_tree, class_label, box_right)

        else:
            leaf_nodes.append(
                LeafNode(
                    tree_id=i_tree,
                    class_label=class_label,
                    node_id=node_id,
                    box=box,
                    value=self.get_values_at_node(node_id)[0, 0],
                )
            )

        return leaf_nodes
