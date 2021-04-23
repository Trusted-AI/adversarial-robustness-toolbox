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
This module implements the task specific estimator for Espresso, an end-to-end speech recognition toolkit based on fairseq.
| Paper link: https://arxiv.org/abs/1909.08723
"""
import logging
from typing import List, Optional, TYPE_CHECKING, Union

import numpy as np

from art.config import ART_NUMPY_DTYPE
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin

from pathlib import Path

if TYPE_CHECKING:
    import torch
    
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE

logger = logging.getLogger(__name__)

INT16MAX = 32767


class PyTorchEspresso(SpeechRecognizerMixin, PyTorchEstimator):
    """
    This class implements a model-specific automatic speech recognizer using the end-to-end speech recognizer
    in Espresso.

    | Paper link: https://arxiv.org/abs/1909.08723
    """

    estimator_params = PyTorchEstimator.estimator_params

    def __init__(
        self,
        espresso_config_filepath: str,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu",
    ):
        """
        Initialization of an instance PyTorchEspresso
        :param espresso_config_filepath: The path of the espresso config file (yaml) # TODO: the config file is not automatically generated in Espresso. It was manually created by us.
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
        import torch  # lgtm [py/repeated-import]
        import yaml
        from argparse import Namespace
        from fairseq import checkpoint_utils, tasks, utils
        from fairseq.data import encoders
        import sentencepiece as spm

        # Super initialization
        super().__init__(
            model=None,
            clip_values=clip_values,
            channels_first=None,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        # Check clip values
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == -1):
                raise ValueError(
                    "This estimator requires normalized input audios with clip_vales=(-1, 1).")
            if not np.all(self.clip_values[1] == 1):
                raise ValueError(
                    "This estimator requires normalized input audios with clip_vales=(-1, 1).")

        # Check postprocessing defences)
        if self.postprocessing_defences is not None:
            raise ValueError(
                "This estimator does not support `postprocessing_defences`.")

        # Set cpu/gpu device
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx)) 

        # construct args
        with open(config_filepath) as file:
            esp_args_dict = yaml.load(file, Loader=yaml.FullLoader)
            esp_args = Namespace(**esp_args_dict)
        self.esp_args = esp_args

        # setup espresso/fairseq task
        self.task = tasks.setup_task(self.esp_args)
        self.task.feat_dim = self.esp_args.feat_dim

        # load_model_ensemble
        self._models, self._model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.esp_args.path),
            arg_overrides=eval(self.esp_args.model_overrides),
            task=self.task,
            suffix=getattr(self.esp_args, "checkpoint_suffix", ""),
        )
        for m in self._models:
            m.to(self._device)

        self._model = self._models[0]

        self.dictionary = self.task.target_dictionary
        self.generator = self.task.build_generator(self._models, self.esp_args)
        self.tokenizer = encoders.build_tokenizer(self.esp_args)
        self.bpe = encoders.build_bpe(self.esp_args) # bpe encoder
        self.sp = spm.SentencePieceProcessor() # sentence piece model
        self.sp.Load(self.esp_args.sentencepiece_model)

        self.criterion = self.task.build_criterion(self.esp_args)

    def predict(
            self, x: np.ndarray, batch_size: int=128, **kwargs) -> np.ndarray:
        """
        Perform prediction for a batch of inputs.
        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param batch_size: Batch size.
        :return: Transcription as a numpy array of characters. A possible example of a transcription return
        is `np.array(['SIXTY ONE', 'HELLO'])`.
        """

        def get_symbols_to_strip_from_output(generator):
            if hasattr(generator, "symbols_to_strip_from_output"):
                return generator.symbols_to_strip_from_output
            else:
                return {generator.eos, generator.pad}

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        # Run prediction with batch processing
        decoded_output = []
        result_output_sizes = np.zeros(x_preprocessed.shape[0], dtype=np.int)
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Transform x into the model input space
            # Note that batch is re-ordered during transformation
            batch, batch_idx = self.transform_model_input(
                x=x_preprocessed[begin:end])

            hypos = self.task.inference_step(
                self.generator, self._models, batch)

            for i in range(len(hypos)):
                # Process top predictions
                for j, hypo in enumerate(hypos[i][:self.esp_args.nbest]):
                    hypo_str = self.dictionary.string(
                        hypo["tokens"].int().cpu(),
                        bpe_symbol=None,
                        extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                            self.generator),
                    )  # not removing bpe at this point
                    detok_hypo_str = self.bpe.decode(hypo_str)
                    decoded_output.append(detok_hypo_str)

        decoded_output = np.array(decoded_output)
        decoded_output_ = decoded_output.copy()
        decoded_output[batch_idx] = decoded_output_ # revert decoded output to its original order
        return decoded_output

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.
        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param y: Target values of shape (nb_samples). Each sample in `y` is a string and it may possess different
                  lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: Loss gradients of the same shape as `x`.
        """

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(
            x, y, fit=False)

        self._model.train()
        self.set_batchnorm(train=False)
        
        # Transform data into the model input space
        batch, batch_idx = self.transform_model_input(
            x=x_preprocessed, y=y_preprocessed, compute_gradient=True)

        loss, sample_size, log_output = self.criterion(self._model, batch)
        loss.backward()

        # Get results
        results = []
        for i in range(len(x_preprocessed)):
            results.append(x_preprocessed[i].grad.cpu().numpy().copy())

        results = np.array(results)
        results = self._apply_preprocessing_gradient(x, results)

        self.set_batchnorm(train=True)
        return results

    def transform_model_input(
            self,
            x: Union[np.ndarray, "torch.Tensor"],
            y: Optional[np.ndarray] = None,
            compute_gradient: bool = False,
            tensor_input: bool = False,
    ) -> Tuple[dict, List]:
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
        :return: A tupe of a dictionary of batch and a list representing the original order of the batch
        """
        import torch  # lgtm [py/repeated-import]
        from fairseq.data import data_utils

        def _collate_fn(batch: List) -> dict:
            """
            Collate function that transforms a list of numpy array or torch tensor representing a batch
            into a dictionary that Espresso takes as input
            """
            # sort by seq length in descending order
            batch = sorted(batch, key=lambda t: t[0].size(0), reverse=True)
            batch_size = len(batch)
            max_seqlength = batch[0][0].size(0)
            src_frames = torch.zeros(batch_size, max_seqlength, 1)
            src_lengths = torch.zeros(batch_size, dtype=torch.long)

            for i, (sample, _) in enumerate(batch):
                seq_length = sample.size(0)
                src_frames[i, :seq_length, :] = sample.unsqueeze(1)
                src_lengths[i] = seq_length

            # for input feeding in training
            if batch[0][1] is not None:
                pad_idx = self.dictionary.pad()
                eos_idx = self.dictionary.eos()
                target = data_utils.collate_tokens(
                    [s[1] for s in batch],
                    pad_idx, eos_idx, False, False,
                    pad_to_length=None,
                    pad_to_multiple=1,
                )
                prev_output_tokens = data_utils.collate_tokens(
                    [s[1] for s in batch],
                    pad_idx, eos_idx, False, True,
                    pad_to_length=None,
                    pad_to_multiple=1,
                )
                target = target.long().to(self._device)
                prev_output_tokens = prev_output_tokens.long().to(self._device)
                ntokens = sum(s[1].ne(pad_idx).int().sum().item()
                              for s in batch)

            else:
                target = None
                prev_output_tokens = None
                ntokens = None

            batch_dict = {
                "ntokens": ntokens,
                "net_input": {
                    "src_tokens": src_frames.to(self._device),
                    "src_lengths": src_lengths.to(self._device),
                    "prev_output_tokens": prev_output_tokens
                },
                "target": target
            }

            return batch_dict

        # We must process each sequence separately due to the diversity of their length
        batch = []
        for i in range(len(x)):
            # First process the target
            if y is None:
                target = None
            else:
                sp = self.sp.EncodeAsPieces(y[i])
                sp_string = " ".join(sp)
                target = self.dictionary.encode_line(
                    sp_string, add_if_not_exist=False)  # target is a long tensor

            # Push the sequence to device
            if not tensor_input:
                x[i] = x[i].astype(ART_NUMPY_DTYPE)
                x[i] = torch.tensor(x[i]).to(self._device)

            # Set gradient computation permission
            if compute_gradient:
                x[i].requires_grad = True

            # Re-scale the input audio to the magnitude used to train Espresso model
            x_i = x[i] * 32767

            # Smoothing comes after WaveGAN but before the quantization
            batch.append((x_i, target))

        # We must keep the order of the batch for later use as the following function will change its order
        batch_idx = sorted(range(len(batch)),
                           key=lambda i: batch[i][0].size(0), reverse=True)

        # The collate function is important to convert input into model space
        batch_dict = _collate_fn(batch)

        return batch_dict, batch_idx

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def model(self) -> "DeepSpeech":
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
