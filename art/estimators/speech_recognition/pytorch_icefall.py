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
This module implements the task specific estimator for Icefall, an end-to-end speech recognition toolkit based on
k2-fsa.

| Repository link: https://github.com/k2-fsa/icefall/tree/master
"""
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import torch

from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin, PytorchSpeechRecognizerMixin

if TYPE_CHECKING:
    # pylint: disable=C0412

    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE

logger = logging.getLogger(__name__)


class PyTorchIcefall(PytorchSpeechRecognizerMixin, SpeechRecognizerMixin, PyTorchEstimator):
    """
    This class implements a model-specific automatic speech recognizer using the end-to-end speech recognizer in
    Icefall.

    | Repository link: https://github.com/k2-fsa/icefall/tree/master
    """

    from icefall.utils import AttributeDict
    from pathlib import Path
    import k2

    estimator_params = PyTorchEstimator.estimator_params

    def __init__(
        self,
        model: Optional[str] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu",
        verbose: bool = True,
        model_ensemble=None,
    ):
        """
        Initialization of an instance PyTorchIcefall

        :param model: The choice of pretrained model if a pretrained model is required.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch
        import yaml

        # Super initialization
        super().__init__(
            model=None,
            clip_values=clip_values,
            channels_first=None,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )
        self.verbose = verbose

        # Check clip values
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == -1):  # pragma: no cover
                raise ValueError("This estimator requires normalized input audios with clip_vales=(-1, 1).")
            if not np.all(self.clip_values[1] == 1):  # pragma: no cover
                raise ValueError("This estimator requires normalized input audios with clip_vales=(-1, 1).")

        # Check postprocessing defences
        if self.postprocessing_defences is not None:  # pragma: no cover
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        # load_model_ensemble
        if model_ensemble is not None:
            self.params = model_ensemble["params"]
            self.transducer_model = model_ensemble["model"]
            self.word2ids = model_ensemble["word2ids"]
            self.get_id2word = model_ensemble["get_id2word"]

        self.transducer_model.to(self.device)

    def predict(self, x: np.ndarray, batch_size: int = 1, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param batch_size: Batch size.
        :return: Transcription as a numpy array of characters. A possible example of a transcription return
                 is `np.array(['SIXTY ONE', 'HELLO'])`.
        """
        from transducer.beam_search import greedy_search

        assert batch_size == 1

        x_in = np.empty(len(x), dtype=object)
        x_in[:] = list(x)
        assert len(x) == 1

        # Put the model in the eval mode
        self.transducer_model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x_in, y=None, fit=False)

        # Run prediction with batch processing
        decoded_output = []
        # result_output_sizes = np.zeros(x_preprocessed.shape[0], dtype=int)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for sample_index in range(num_batch):
            wav = x_preprocessed[sample_index]  # np.array, len = wav len

            # extract features
            x, _, _ = self.transform_model_input(x=torch.tensor(wav))
            shape = torch.tensor([x.shape[1]])

            encoder_out, encoder_out_lens = self.transducer_model.encoder(x=x, x_lens=shape)
            hyp = greedy_search(model=self.transducer_model, encoder_out=encoder_out, id2word=self.get_id2word)
            decoded_output.append(hyp)

        return np.concatenate(decoded_output)

    def loss_gradient(self, x, y: np.ndarray, **kwargs) -> np.ndarray:
        import k2

        x = torch.autograd.Variable(x, requires_grad=True)
        features, _, _ = self.transform_model_input(x=x, compute_gradient=True)
        x_lens = torch.tensor([features.shape[1]]).to(torch.int32).to(self.device)
        y = k2.RaggedTensor(y)
        loss = self.transducer_model(x=features, x_lens=x_lens, y=y)
        loss.backward()

        # Get results
        results = x.grad
        results = self._apply_preprocessing_gradient(x, results)
        return results

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128, nb_epochs: int = 10, **kwargs) -> None:
        """
        Fit the estimator on the training set `(x, y)`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
                       and providing it takes no effect.
        """
        raise NotImplementedError

    def transform_model_input(self, x, y=None, compute_gradient=False):
        """
        Transform the user input space into the model input space.
        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param compute_gradient: Indicate whether to compute gradients for the input `x`.
        :param tensor_input: Indicate whether input is tensor.
        :param real_lengths: Real lengths of original sequences.
        :return: A tupe of a sorted input feature tensor, a supervision tensor,  and a list representing the original order of the batch
        """
        import torch  # lgtm [py/repeated-import]
        import torchaudio

        from dataclasses import dataclass, asdict

        params = {
            # Spectogram-related part
            "dither": 0.0,
            "window_type": "povey",
            "sample_frequency": 16000,
            "snip_edges": False,
            "num_mel_bins": 23,
            # Note that frame_length and frame_shift will be converted to milliseconds before torchaudio/Kaldi sees them
            "frame_length": 25.0,
            "frame_shift": 10.0,
            "remove_dc_offset": True,
            "round_to_power_of_two": True,
            "energy_floor": 1e-10,
            "min_duration": 0.0,
            "preemphasis_coefficient": 0.97,
            "raw_energy": True,
            # Fbank-related part
            "low_freq": 20.0,
            "high_freq": -400.0,
            "num_mel_bins": 40,
            "use_energy": False,
            "vtln_low: float": 100.0,
            "vtln_high: float": -500.0,
            "vtln_warp: float": 1.0,
        }

        feature_list = []
        num_frames = []
        supervisions = {}

        for i in range(len(x)):
            isnan = torch.isnan(x[i])
            nisnan = torch.sum(isnan).item()
            if nisnan > 0:
                logging.info(f"input isnan={nisnan}/{x[i][isnan]} {torch.max(torch.abs(x[i]))}")

            xx = x[i]
            xx = xx.to(self._device)
            feat_i = torchaudio.compliance.kaldi.fbank(xx.unsqueeze(0), **params)  # [T, C]
            feat_i = feat_i.transpose(0, 1)  # [C, T]
            feature_list.append(feat_i)
            num_frames.append(feat_i.shape[1])

        indices = sorted(range(len(feature_list)), key=lambda i: feature_list[i].shape[1], reverse=True)
        indices = torch.tensor(indices, dtype=torch.int64, device=self._device)
        num_frames = torch.tensor([num_frames[idx] for idx in indices], dtype=torch.int32, device=self._device)
        start_frames = torch.zeros(len(x), dtype=torch.int32, device=self._device)

        supervisions["sequence_idx"] = indices.int()
        supervisions["start_frame"] = start_frames
        supervisions["num_frames"] = num_frames
        if y is not None:
            supervisions["text"] = [y[idx] for idx in indices]

        feature_sorted = [feature_list[index] for index in indices]

        feature = torch.zeros(
            len(feature_sorted), feature_sorted[0].size(0), feature_sorted[0].size(1), device=self._device
        )

        for i in range(len(x)):
            feature[i, :, : feature_sorted[i].size(1)] = feature_sorted[i]

        return feature.transpose(1, 2), supervisions, indices

    def to_training_mode(self) -> None:
        """
        Put the estimator in the training mode.
        """
        self.transducer_model.train()

    @property
    def sample_rate(self) -> int:
        """
        Get the sampling rate.

        :return: The audio sampling rate.
        """
        return self._sampling_rate

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def model(self) -> "torch.nn.Module":
        """
        Get current model.

        :return: Current model.
        """
        return self._model

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    def compute_loss_and_decoded_output(
        self, masked_adv_input: "torch.Tensor", original_output: np.ndarray, **kwargs
    ) -> Tuple["torch.Tensor", np.ndarray]:
        """
        Compute loss function and decoded output.

        :param masked_adv_input: The perturbed inputs.
        :param original_output: Target values of shape (nb_samples). Each sample in `original_output` is a string and
                                it may possess different lengths. A possible example of `original_output` could be:
                                `original_output = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: The loss and the decoded output.
        """
        from transducer.beam_search import greedy_search
        import k2

        assert len(original_output[0]) == 1
        num_batch = len(original_output[0])
        decoded_output = []

        for sample_index in range(num_batch):
            features, _, _ = self.transform_model_input(x=masked_adv_input[sample_index])
            x_lens = torch.tensor([features.shape[1]]).to(torch.int32).to(self.device)
            y = k2.RaggedTensor(original_output[sample_index])
            loss = self.transducer_model(x=features, x_lens=x_lens, y=y)

            encoder_out, encoder_out_lens = self.transducer_model.encoder(
                x=features, x_lens=masked_adv_input[sample_index].shape
            )
            hyp = greedy_search(model=self.transducermodel, encoder_out=encoder_out, id2word=self.get_id2word)
            decoded_output.append(hyp)

        return loss, np.concatenate(decoded_output)
