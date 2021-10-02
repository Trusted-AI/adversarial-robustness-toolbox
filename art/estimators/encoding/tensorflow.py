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
This module implements the classifier `TensorFlowEncoder` for TensorFlow models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING

from art.estimators.encoding.encoder import EncoderMixin
from art.estimators.tensorflow import TensorFlowEstimator

if TYPE_CHECKING:
    # pylint: disable=C0412
    import numpy as np
    import tensorflow.compat.v1 as tf

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class TensorFlowEncoder(EncoderMixin, TensorFlowEstimator):  # lgtm [py/missing-call-to-init]
    """
    This class implements an encoder model using the TensorFlow framework.
    """

    estimator_params = TensorFlowEstimator.estimator_params + [
        "input_ph",
        "loss",
        "sess",
        "feed_dict",
        "channels_first",
    ]

    def __init__(
        self,
        input_ph: "tf.Placeholder",
        model: "tf.Tensor",
        loss: Optional["tf.Tensor"] = None,
        sess: Optional["tf.compat.v1.Session"] = None,
        channels_first: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        feed_dict: Optional[Dict[Any, Any]] = None,
    ):
        """
        Initialization specific to encoder estimator implementation in TensorFlow.

        :param input_ph: The input placeholder.
        :param model: TensorFlow model, neural network or other.
        :param loss: The loss function for which to compute gradients. This parameter is necessary when training the
                     model and when computing gradients w.r.t. the loss function.
        :param sess: Computation session.
        :param channels_first: Set channels first or last.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
                            maximum values allowed for features. If floats are provided, these will be used as the range
                            of all features. If arrays are provided, each value will be considered the bound for a
                            feature, thus the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
                              used for data preprocessing. The first value will be subtracted from the input. The input
                              will then be divided by the second one.
        :param feed_dict: A feed dictionary for the session run evaluating the classifier. This dictionary includes all
                          additionally required placeholders except the placeholders defined in this class.
        """
        import tensorflow.compat.v1 as tf  # lgtm [py/repeated-import]

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._nb_classes = int(model.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape().as_list()[1:])
        self._input_ph = input_ph
        self._encoding_length = self._model.shape[1]
        self._loss = loss
        if feed_dict is None:
            self._feed_dict = dict()
        else:
            self._feed_dict = feed_dict

        # Assign session
        if sess is None:  # pragma: no cover
            raise ValueError("A session cannot be None.")
        self._sess = sess

        # Get the loss gradients graph
        if self.loss is not None:
            self._loss_grads = tf.gradients(self.loss, self.input_ph)[0]

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def input_ph(self) -> "tf.Placeholder":
        """
        Return the input placeholder.

        :return: The input placeholder.
        """
        return self._input_ph  # type: ignore

    @property
    def loss(self) -> "tf.Tensor":
        """
        Return the loss function.

        :return: The loss function.
        """
        return self._loss  # type: ignore

    @property
    def feed_dict(self) -> Dict[Any, Any]:
        """
        Return the feed dictionary for the session run evaluating the classifier.

        :return: The feed dictionary for the session run evaluating the classifier.
        """
        return self._feed_dict  # type: ignore

    def predict(self, x: "np.ndarray", batch_size: int = 128, **kwargs):
        """
        Perform prediction for a batch of inputs.

        :param x: Input samples.
        :param batch_size: Batch size.
        :return: Array of encoding predictions of shape `(num_inputs, encoding_length)`.
        """
        logger.info("Encoding input")
        feed_dict = {self.input_ph: x}
        if self.feed_dict is not None:
            feed_dict.update(self.feed_dict)
        y = self._sess.run(self._model, feed_dict=feed_dict)
        return y

    def fit(self, x: "np.ndarray", y: "np.ndarray", batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
        Do nothing.
        """
        raise NotImplementedError

    def get_activations(
        self, x: "np.ndarray", layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> "np.ndarray":
        """
        Do nothing.
        """
        raise NotImplementedError

    def compute_loss(self, x: "np.ndarray", y: "np.ndarray", **kwargs) -> "np.ndarray":
        raise NotImplementedError

    def loss_gradient(self, x: "np.ndarray", y: "np.ndarray", **kwargs) -> "np.ndarray":  # pylint: disable=W0221
        """
        No gradients to compute for this method; do nothing.
        """
        raise NotImplementedError

    @property
    def encoding_length(self) -> int:
        """
        Returns the length of the encoding size output.

        :return: The length of the encoding size output.
        """
        return self._encoding_length
