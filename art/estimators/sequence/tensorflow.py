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
This module implements task-specific estimators for automatic speech recognition in TensorFlow.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np

from art.config import ART_DATA_PATH
from art.estimators.sequence.sequence import SequenceNetworkMixin
from art.estimators.tensorflow import TensorFlowV2Estimator
from art.utils import get_file, make_directory

if TYPE_CHECKING:
    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

    from tensorflow.compat.v1 import Tensor

logger = logging.getLogger(__name__)


class LingvoAsr(SequenceNetworkMixin, TensorFlowV2Estimator):
    """
    This class implements the task-specific Lingvo ASR model of Qin et al. (2019).

    | Paper link: http://proceedings.mlr.press/v97/qin19a.html
    |             https://arxiv.org/abs/1902.08295
    """

    _LINGVO_CFG = {
        "path": os.path.join(ART_DATA_PATH, "lingvo/"),
        "model_ckpt_data_uri": "http://cseweb.ucsd.edu/~yaq007/ckpt-00908156.data-00000-of-00001",
        "model_ckpt_index_uri": (
            "https://github.com/tensorflow/cleverhans/blob/"
            "6ef939059172901db582c7702eb803b7171e3db5/examples/adversarial_asr/model/ckpt-00908156.index?raw=true"
        ),
        "params_uri": (
            "https://raw.githubusercontent.com/tensorflow/lingvo/"
            "3c5ef88b8a9407124afe045a8e5048a9c5013acd/lingvo/tasks/asr/params/librispeech.py"
        ),
        "vocab_uri": (
            "https://raw.githubusercontent.com/tensorflow/lingvo/"
            "3c5ef88b8a9407124afe045a8e5048a9c5013acd/lingvo/tasks/asr/wpm_16k_librispeech.vocab"
        ),
    }

    def __init__(
        self,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "cpu",
    ):
        """
        Initialization.

        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import tensorflow.compat.v1 as tf1

        # Super initialization
        super().__init__(
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.device_type = device_type
        if self.preprocessing is not None:
            raise ValueError("This estimator does not support `preprocessing`.")
        if self.preprocessing_defences is not None:
            raise ValueError("This estimator does not support `preprocessing_defences`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")
        if self.device_type != "cpu":
            raise ValueError("This estimator does not yet support running on a GPU.")

        # disable eager execution as Lingvo uses tensorflow.compat.v1 API
        tf1.disable_eager_execution()

        # init necessary local Lingvo ASR namespace and flags
        sys.path.append(self._LINGVO_CFG["path"])
        tf1.flags.FLAGS(tuple(sys.argv[0]))

        # check and download additional Lingvo ASR params file
        _ = self._check_and_download_params()

        # placeholders
        self._x_padded = tf1.placeholder(tf1.float32, shape=[None, None], name="art_x_padded")

        # init Lingvo computation graph
        self._sess = tf1.Session()
        model, task = self._load_model()
        self._model = model
        self._task = task

    def _check_and_download_params(self) -> str:
        """Check and download the `params/librispeech.py` file from the official Lingvo repository."""
        params_dir = os.path.join(self._LINGVO_CFG["path"], "asr")
        params_base = "librispeech.py"
        if not os.path.isdir(params_dir):
            make_directory(params_dir)
        if not os.path.isfile(os.path.join(params_dir, params_base)):
            logger.info("Could not find %s. Downloading it now...", params_base)
            get_file(params_base, self._LINGVO_CFG["params_uri"], path=params_dir)
        return os.path.join(params_dir, params_base)

    def _check_and_download_model(self) -> Tuple[str, str]:
        """Check and download the pretrained model of Qin et al. (2019). file from the official Lingvo repository."""
        model_ckpt_dir = os.path.join(self._LINGVO_CFG["path"], "asr", "model")
        model_ckpt_data_base = "ckpt-00908156.data-00000-of-00001"
        model_ckpt_index_base = "ckpt-00908156.index"
        if not os.path.isdir(model_ckpt_dir):
            make_directory(model_ckpt_dir)
        if not os.path.isfile(os.path.join(model_ckpt_dir, model_ckpt_data_base)):
            logger.info("Could not find %s. Downloading it now...", model_ckpt_data_base)
            get_file(model_ckpt_data_base, self._LINGVO_CFG["model_ckpt_data_uri"], path=model_ckpt_dir)
        if not os.path.isfile(os.path.join(model_ckpt_dir, model_ckpt_index_base)):
            logger.info("Could not find %s. Downloading it now...", model_ckpt_index_base)
            get_file(model_ckpt_index_base, self._LINGVO_CFG["model_ckpt_index_uri"], path=model_ckpt_dir)
        return os.path.join(model_ckpt_dir, model_ckpt_data_base), os.path.join(model_ckpt_dir, model_ckpt_index_base)

    def _check_and_download_vocab(self) -> str:
        """Check and download the `wpm_16k_librispeech.vocab` file from the official Lingvo repository."""
        vocab_dir = os.path.join(self._LINGVO_CFG["path"], "asr")
        vocab_base = "wpm_16k_librispeech.vocab"
        if not os.path.isdir(vocab_dir):
            make_directory(vocab_dir)
        if not os.path.isfile(os.path.join(vocab_dir, vocab_base)):
            logger.info("Could not find %x. Downloading it now...", vocab_base)
            get_file(vocab_base, self._LINGVO_CFG["vocab_uri"], path=vocab_dir)
        return os.path.join(vocab_dir, vocab_base)

    def _load_model(self):
        """
        Define and instantiate the computation graph.
        """
        import tensorflow.compat.v1 as tf1
        from lingvo import model_registry, model_imports
        from lingvo.core import cluster_factory

        from asr.librispeech import Librispeech960Wpm

        # check and download Lingvo ASR vocab
        vocab_path = self._check_and_download_vocab()

        # monkey-patch tasks.asr.librispeechLibriSpeech960Wpm class attribute WPM_SYMBOL_TABLE_FILEPATH
        Librispeech960Wpm.WPM_SYMBOL_TABLE_FILEPATH = vocab_path

        # register model params
        model_name = "asr.librispeech.Librispeech960Wpm"
        model_imports.ImportParams(model_name)
        params = model_registry._ModelRegistryHelper.GetParams(model_name, "Test")

        # instantiate Lingvo ASR model
        cluster = cluster_factory.Cluster(params.cluster)
        with cluster, tf1.device(cluster.GetPlacer()):
            model = params.Instantiate()
            task = model.GetTask()

        # load Qin et al. pretrained model
        _, model_index_path = self._check_and_download_model()
        self._sess.run(tf1.global_variables_initializer())
        saver = tf1.train.Saver([var for var in tf1.global_variables() if var.name.startswith("librispeech")])
        saver.restore(self._sess, os.path.splitext(model_index_path)[0])

        return model, task

    def _create_log_mel_features(self, x: "Tensor") -> "Tensor":
        """Extract Log-Mel features from audio samples of shape (batch_size, max_length)."""
        from lingvo.core.py_utils import NestedMap
        import tensorflow.compat.v1 as tf1

        def _create_asr_frontend():
            """Parameters corresponding to default ASR frontend."""
            from lingvo.tasks.asr import frontend

            params = frontend.MelAsrFrontend.Params()
            # default params from Lingvo
            params.sample_rate = 16000.0
            params.frame_size_ms = 25.0
            params.frame_step_ms = 10.0
            params.num_bins = 80
            params.lower_edge_hertz = 125.0
            params.upper_edge_hertz = 7600.0
            params.preemph = 0.97
            params.noise_scale = 0.0
            params.pad_end = False
            # additional params from Qin et al.
            params.stack_left_context = 2
            params.frame_stride = 3
            return params.Instantiate()

        # init Lingvo ASR frontend
        mel_frontend = _create_asr_frontend()

        # extract log-mel features
        log_mel = mel_frontend.FPropDefaultTheta(NestedMap(src_inputs=x, paddings=tf1.zeros_like(x)))
        features = log_mel.src_inputs

        # reshape features to shape (batch_size, n_frames, n_features, channels) in compliance with Qin et al.
        features_shape = (tf1.shape(x)[0], -1, 80, features.shape[-1])
        features = tf1.reshape(features, features_shape)
        return features

    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        :return: Loss gradients w.r.t. `x` in the same format as `x`.
        :rtype: Format as expected by the `model`
        """
        raise NotImplementedError

    def set_learning_phase(self) -> None:
        raise NotImplementedError

    def get_activations(self) -> np.ndarray:
        raise NotImplementedError
