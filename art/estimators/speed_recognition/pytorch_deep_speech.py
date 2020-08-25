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
This module implements the task specific estimator for DeepSpeech, an end-to-end speech recognition in English and
Mandarin in PyTorch.

| Paper link: https://arxiv.org/abs/1512.02595
"""
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from deepspeech_pytorch.utils import load_model

from art.estimators.speed_recognition.speed_recognizer import SpeedRecognizerMixin
from art.estimators.pytorch import PyTorchEstimator
from art.utils import get_file
from art.config import ART_DATA_PATH, ART_NUMPY_DTYPE

if TYPE_CHECKING:
    from deepspeech_pytorch.model import DeepSpeech

    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchDeepSpeech(SpeedRecognizerMixin, PyTorchEstimator):
    """
    This class implements a model-specific automatic speed recognizer using the end-to-end speech recognizer
    DeepSpeech and PyTorch.

    | Paper link: https://arxiv.org/abs/1512.02595
    """

    def __init__(
        self,
        model: Optional["DeepSpeech"] = None,
        pretrained_model: Optional[str] = None,
        filename: Optional[str] = None,
        url: Optional[str] = None,
        use_half: bool = False,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu"
    ):
        """
        Initialization of an instance PyTorchDeepSpeech.

        :param model: DeepSpeech model.
        :param pretrained_model: The choice of pretrained model if a pretrained model is required. Currently this
                                 estimator supports 3 different pretrained models consisting of `an4`, `librispeech`
                                 and `tedlium`.
        :param filename: Name of the file.
        :param url: Download URL.
        :param use_half: Whether to use FP16 for pretrained model.
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

        # Super initialization
        super().__init__(
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing
        )

        # Check clip values
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == -1):
                raise ValueError("This estimator requires normalized input audios with clip_vales=(-1, 1).")
            if not np.all(self.clip_values[1] == 1):
                raise ValueError("This estimator requires normalized input audios with clip_vales=(-1, 1).")

        # Check preprocessing and postprocessing defences
        if self.preprocessing_defences is not None:
            raise ValueError("This estimator does not support `preprocessing_defences`.")
        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        # Set cpu/gpu device
        self._device: str
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        # Load model
        if model is None:
            if pretrained_model == 'an4':
                filename, url = (
                    "an4_pretrained_v2.pth",
                    "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/an4_pretrained_v2.pth"
                )
            elif pretrained_model == 'librispeech':
                filename, url = (
                    "librispeech_pretrained_v2.pth",
                    "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/"
                    "librispeech_pretrained_v2.pth"
                )
            elif pretrained_model == 'tedlium':
                filename, url = (
                    "ted_pretrained_v2.pth",
                    "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/ted_pretrained_v2.pth"
                )
            elif pretrained_model is None:
                # If model is None and no pretrained model is selected, then we need to have parameters filename and
                # url to download, extract and load the automatic speed recognition model
                if filename is None or url is None:
                    filename, url = (
                        "librispeech_pretrained_v2.pth",
                        "https://github.com/SeanNaren/deepspeech.pytorch/releases/download/v2.0/"
                        "librispeech_pretrained_v2.pth"
                    )
            else:
                raise ValueError("The input pretrained model %s is not supported." % pretrained_model)

            # Download model
            model_path = get_file(filename=filename, path=ART_DATA_PATH, url=url, extract=False)

            # Then load model
            self._model = load_model(device=self._device, model_path=model_path, use_half=use_half)

        else:
            self._model = model

        # Push model to the corresponding device
        self._model.to(self._device)

    def predict(
        self, x: np.ndarray, batch_size: int = 128, **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], List[List[str]]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param batch_size: Batch size.
        :param transcription_output: Indicate whether the function will produce probability or transcription as
                                     prediction output.
        :type transcription_output: `bool`
        :param decoder_type: Decoder type. Either `greedy` or `beam`. This parameter is only used when users want
                             transcription outputs. Default to `greedy`.
        :type decoder_type: `str`
        :param lm_path: Path to an (optional) kenlm language model for use with beam search. This parameter is only
                        used when users want transcription outputs. Default to `''`.
        :type lm_path: `str`
        :param top_paths: Number of beams to return. This parameter is only used when users want transcription outputs.
                          Default to 1.
        :type top_paths: `int`
        :param alpha: Language model weight. This parameter is only used when users want transcription outputs.
                      Default to 0.0.
        :type alpha: `float`
        :param beta: Language model word bonus (all words). This parameter is only used when users want transcription
                     outputs. Default to 0.0.
        :type beta: `float`
        :param cutoff_top_n: Cutoff_top_n characters with highest probs in vocabulary will be used in beam search. This
                             parameter is only used when users want transcription outputs. Default to 40.
        :type cutoff_top_n: `float`
        :param cutoff_prob: Cutoff probability in pruning. This parameter is only used when users want transcription
                            outputs. Default to 1.0.
        :type cutoff_prob: `float`
        :param beam_width: Beam width to use. This parameter is only used when users want transcription outputs.
                           Default to 10.
        :type beam_width: `int`
        :param lm_workers: Number of language model processes to use. This parameter is only used when users want
                           transcription outputs. Default to 4.
        :type lm_workers: `int`
        :return: Probability (if transcription_output is None or False) or transcription (if transcription_output is
                 True) predictions:
                    - Probability return is a tuple of (probs, sizes), where:
                        - probs is the probability of characters of shape (nb_samples, seq_length, nb_classes).
                        - sizes is the real sequence length of shape (nb_samples,).
                    - Transcription return is a numpy array of characters. A possible example of a transcription return
                      is `np.array(['SIXTY ONE', 'HELLO'])`.
        """
        import torch  # lgtm [py/repeated-import]

        # Put the model in the evaluation status
        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Transform x into the model input space
        inputs, targets, input_rates, target_sizes, batch_idx = self._transform_model_input(x=x_preprocessed)

        # Compute real input sizes
        input_sizes = input_rates.mul_(inputs.size(-1)).int()

        # Run prediction with batch processing
        results = []
        result_output_sizes = np.zeros(x_preprocessed.shape[0], dtype=np.int)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Call to DeepSpeech model for prediction
            with torch.no_grad():
                outputs, output_sizes = self._model(inputs[begin : end], input_sizes[begin : end])

            results.append(outputs)
            result_output_sizes[begin : end] = output_sizes.detach().cpu().numpy()

        # Aggregate results
        result_outputs = np.zeros(
            (x_preprocessed.shape[0], result_output_sizes.max(), results[0].shape[-1]), dtype=np.float32
        )
        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Overwrite results
            result_outputs[begin : end, : results[m].shape[1], : results[m].shape[-1]] = results[m]

        # Rearrange to the original order
        result_output_sizes_ = result_output_sizes.copy()
        result_outputs_ = result_outputs.copy()
        result_output_sizes[batch_idx] = result_output_sizes_
        result_outputs[batch_idx] = result_outputs_

        # Check if users want transcription outputs
        transcription_output = kwargs.get("transcription_output")

        if transcription_output is None or transcription_output == False:
            return result_outputs, result_output_sizes

        # Now users want transcription outputs
        # Hence first create a decoder
        from deepspeech_pytorch.configs.inference_config import LMConfig
        from deepspeech_pytorch.enums import DecoderType
        from deepspeech_pytorch.utils import load_decoder

        # Create the language model config
        lm_config = LMConfig()

        decoder_type = kwargs.get("decoder_type")
        if decoder_type is not None:
            if decoder_type == 'greedy':
                lm_config.decoder_type = DecoderType.greedy
            elif decoder_type == 'beam':
                lm_config.decoder_type = DecoderType.beam
            else:
                raise ValueError("Decoder type %s currently not supported." % decoder_type)

        lm_path = kwargs.get("lm_path")
        if lm_path is not None:
            lm_config.lm_path = lm_path

        top_paths = kwargs.get("top_paths")
        if top_paths is not None:
            lm_config.top_paths = top_paths

        alpha = kwargs.get("alpha")
        if alpha is not None:
            lm_config.alpha = alpha

        beta = kwargs.get("beta")
        if beta is not None:
            lm_config.beta = beta

        cutoff_top_n = kwargs.get("cutoff_top_n")
        if cutoff_top_n is not None:
            lm_config.cutoff_top_n = cutoff_top_n

        cutoff_prob = kwargs.get("cutoff_prob")
        if cutoff_prob is not None:
            lm_config.cutoff_prob = cutoff_prob

        beam_width = kwargs.get("beam_width")
        if beam_width is not None:
            lm_config.beam_width = beam_width

        lm_workers = kwargs.get("lm_workers")
        if lm_workers is not None:
            lm_config.lm_workers = lm_workers

        # Create the decoder with the lm config
        decoder = load_decoder(labels=self._model.labels, cfg=lm_config)

        # Compute transcription
        decoded_output, _ = decoder.decode(
            torch.tensor(result_outputs, device=self._device), torch.tensor(result_output_sizes, device=self._device)
        )
        decoded_output = [do[0] for do in decoded_output]
        decoded_output = np.array(decoded_output)

        return decoded_output

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """
        # Put the model in the evaluation status
        self._model.eval()

        # TODO

    def _transform_model_input(
        self, x: np.ndarray, y: Optional[np.ndarray] = None, compute_gradient: bool = False
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", List]:
        """
        Transform the user input space into the model input space.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.ndarray([[0.1, 0.2, 0.1, 0.4], [0.3, 0.1]])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :param compute_gradient: Indicate whether to compute gradients for the input `x`.
        :return: A tuple of inputs and targets in the model space with the original index
                 `(inputs, targets, input_percentages, target_sizes, batch_idx)`, where:
                    - inputs: model inputs of shape (nb_samples, nb_frequencies, seq_length).
                    - targets: ground truth targets of shape (sum over nb_samples of real seq_lengths).
                    - input_percentages: percentages of real inputs in inputs.
                    - target_sizes: list of real seq_lengths.
                    - batch_idx: original index of inputs.
        """
        import torch
        import torchaudio

        from deepspeech_pytorch.loader.data_loader import _collate_fn

        x = x.astype(ART_NUMPY_DTYPE)

        # These parameters are needed for the transformation
        sample_rate = self._model.audio_conf.sample_rate
        window_size = self._model.audio_conf.window_size
        window_stride = self._model.audio_conf.window_stride

        n_fft = int(sample_rate * window_size)
        hop_length = int(sample_rate * window_stride)
        win_length = n_fft

        window = self._model.audio_conf.window.value
        if window == 'hamming':
            window_fn = torch.hamming_window
        elif window == 'hann':
            window_fn = torch.hann_window
        elif window == 'blackman':
            window_fn = torch.blackman_window
        elif window == 'bartlett':
            window_fn = torch.bartlett_window
        else:
            raise NotImplementedError("Spectrogram window %s not supported." % window)

        # Create a transformer to transform between the two spaces
        transformer = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window_fn=window_fn,
            power = None
        )

        # Create a label map
        label_map = dict([(self._model.labels[i], i) for i in range(len(self._model.labels))])

        # We must process each sequence separately due to the diversity of their length
        batch = []
        for i in range(len(x)):
            # First process the target
            if y is None:
                target = []
            else:
                target = list(filter(None, [label_map.get(letter) for letter in list(y[i])]))

            # Push the sequence to device
            x[i] = torch.tensor(x[i]).to(self._device)

            # Set gradient computation permission
            if compute_gradient:
                x[i].requires_grad = True

            # Transform the sequence into the frequency space
            transformed_input = transformer(x[i])
            spectrogram, _ = torchaudio.functional.magphase(transformed_input)
            spectrogram = torch.log1p(spectrogram)

            # Normalize data
            mean = spectrogram.mean()
            std = spectrogram.std()
            spectrogram = spectrogram - mean
            spectrogram = spectrogram / std

            # Then form the batch
            batch.append((spectrogram, target))

        # We must keep the order of the batch for later use as the following function will change its order
        batch_idx = sorted(range(len(batch)), key=lambda i: batch[i][0].size(1), reverse=True)

        # The collate function is important to convert input into model space
        inputs, targets, input_percentages, target_sizes = _collate_fn(batch)

        return inputs, targets, input_percentages, target_sizes, batch_idx



    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        # Put the model in the training status
        self._model.train()

        # TODO


    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def set_learning_phase(self, train: bool) -> None:
        raise NotImplementedError
