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

from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.pytorch import PyTorchEstimator
from art.utils import get_file
from art.config import ART_DATA_PATH, ART_NUMPY_DTYPE

if TYPE_CHECKING:
    import torch

    from deepspeech_pytorch.model import DeepSpeech

    from art.config import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchEspresso(SpeechRecognizerMixin, PyTorchEstimator):
    """
    This class implements a model-specific automatic speech recognizer using the end-to-end speech recognizer
    in Espresso.
    """

    def __init__(
            self,
            infer_args,
            clip_values: Optional["CLIP_VALUES_TYPE"] = None,
            preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
            postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
            preprocessing: "PREPROCESSING_TYPE" = None,
            device_type: str = "cpu",
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
        :param optimizer: The optimizer used to train the estimator.
        :param use_amp: Whether to use the automatic mixed precision tool to enable mixed precision training or
                        gradient computation, e.g. with loss gradient computation. When set to True, this option is
                        only triggered if there are GPUs available.
        :param opt_level: Specify a pure or mixed precision optimization level. Used when use_amp is True. Accepted
                          values are `O0`, `O1`, `O2`, and `O3`.
        :param loss_scale: Loss scaling. Used when use_amp is True. Default is 1.0 due to warp-ctc not supporting
                           scaling of gradients. If passed as a string, must be a string representing a number,
                           e.g., “1.0”, or the string “dynamic”.
        :param decoder_type: Decoder type. Either `greedy` or `beam`. This parameter is only used when users want
                             transcription outputs.
        :param lm_path: Path to an (optional) kenlm language model for use with beam search. This parameter is only
                        used when users want transcription outputs.
        :param top_paths: Number of beams to be returned. This parameter is only used when users want transcription
                          outputs.
        :param alpha: The weight used for the language model. This parameter is only used when users want transcription
                      outputs.
        :param beta: Language model word bonus (all words). This parameter is only used when users want transcription
                     outputs.
        :param cutoff_top_n: Cutoff_top_n characters with highest probs in vocabulary will be used in beam search. This
                             parameter is only used when users want transcription outputs.
        :param cutoff_prob: Cutoff probability in pruning. This parameter is only used when users want transcription
                            outputs.
        :param beam_width: The width of beam to be used. This parameter is only used when users want transcription
                           outputs.
        :param lm_workers: Number of language model processes to use. This parameter is only used when users want
                           transcription outputs.
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

        from fairseq import checkpoint_utils, options, tasks, utils
        from fairseq.data import encoders
        import sentencepiece as spm

        # Super initialization
        super().__init__(
            clip_values=clip_values,
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

        # Check preprocessing and postprocessing defences
        if self.preprocessing_defences is not None:
            raise ValueError(
                "This estimator does not support `preprocessing_defences`.")
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

        # # Save first version of the optimizer
        # self._optimizer = optimizer
        # self._use_amp = use_amp

        # construct args
        self.infer_args = infer_args

        # setup task
        self.infer_task = tasks.setup_task(self.infer_args)
        self.infer_task.feat_dim = 83

        # load_model_ensemble
        self.models, self._model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(self.infer_args.path),
            arg_overrides=eval(self.infer_args.model_overrides),
            task=self.infer_task,
            suffix=getattr(self.infer_args, "checkpoint_suffix", ""),
        )

        for m in self.models:
            m.to(self._device)
        
        self.dictionary = self.infer_task.target_dictionary
        self.generator = self.infer_task.build_generator(self.models, self.infer_args)
        self.tokenizer = encoders.build_tokenizer(self.infer_args)
        self.bpe = encoders.build_bpe(self.infer_args)
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.infer_args.sentencepiece_model)

    def predict(
        self, x: np.ndarray, batch_size: int = 128, **kwargs
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, seq_length). Note that, it is allowable that sequences in the batch
                  could have different lengths. A possible example of `x` could be:
                  `x = np.array([np.array([0.1, 0.2, 0.1, 0.4]), np.array([0.3, 0.1])])`.
        :param batch_size: Batch size.
        :param transcription_output: Indicate whether the function will produce probability or transcription as
                                     prediction output. If transcription_output is not available, then probability
                                     output is returned.
        :type transcription_output: `bool`
        :return: Probability (if transcription_output is None or False) or transcription (if transcription_output is
                 True) predictions:
                 - Probability return is a tuple of (probs, sizes), where `probs` is the probability of characters of
                 shape (nb_samples, seq_length, nb_classes) and `sizes` is the real sequence length of shape
                 (nb_samples,).
                 - Transcription return is a numpy array of characters. A possible example of a transcription return
                 is `np.array(['SIXTY ONE', 'HELLO'])`.
        """
        import torch  # lgtm [py/repeated-import]

        x_ = np.array([x_i for x_i in x] +
                      [np.array([0.1]), np.array([0.1, 0.2])])[:-2]

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x_, y=None, fit=False)

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

            # Transform x into the model input space
            batch, batch_idx = self.transform_model_input(
                x=x_preprocessed[begin:end])

            hypos = self.task.inference_step(
                self.generator, self.models, batch)

            outputs = hypos["?"]
            results.append(outputs)
            result_output_sizes[begin:end] = output_sizes.detach(
            ).cpu().numpy()

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
            result_outputs[begin:end, : results[m].shape[1],
                           : results[m].shape[-1]] = results[m].cpu().numpy()

        # Rearrange to the original order
        result_output_sizes_ = result_output_sizes.copy()
        result_outputs_ = result_outputs.copy()
        result_output_sizes[batch_idx] = result_output_sizes_
        result_outputs[batch_idx] = result_outputs_

        # Check if users want transcription outputs
        transcription_output = kwargs.get("transcription_output")

        if transcription_output is None or transcription_output is False:
            return result_outputs, result_output_sizes

        # Now users want transcription outputs
        decoded_output = []
        for i in range(batch_size):
            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args.nbest]):
                hypo_str = self.dictionary.string(
                    hypo["tokens"].int().cpu(),
                    bpe_symbol=None,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                        self.generator),
                )  # not removing bpe at this point
                detok_hypo_str = self.tokenizer(hypo_str)
                decoded_output.append(detok_hypo_str)
        decoded_output = np.array(decoded_output)

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

        x_ = np.array([x_i for x_i in x] +
                      [np.array([0.1]), np.array([0.1, 0.2])])[:-2]

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(
            x_, y, fit=False)

        # Transform data into the model input space
        inputs, targets, input_rates, target_sizes, batch_idx = self.transform_model_input(
            x=x_preprocessed, y=y_preprocessed, compute_gradient=True
        )

        # Transform x into the model input space
        batch, batch_idx = self.transform_model_input(
            x=x_preprocessed[begin:end])
        loss, sample_size_i, logging_output = self.task.train_step(
            sample=batch,
            model=self.model,
            criterion=self.criterion,
            optimizer=self._optimizer,
        )

        # Get results
        results = []
        for i in range(len(x_preprocessed)):
            results.append(x_preprocessed[i].grad.cpu().numpy().copy())

        results = np.array(results)
        results = self._apply_preprocessing_gradient(x_, results)

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
        import random

        if self._optimizer is None:
            raise ValueError(
                "An optimizer is required to train the model, but none was provided.")

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(
            x, y, fit=True)

        # Train with batch processing
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        ind = np.arange(len(x_preprocessed))

        # Loss function
        criterion = CTCLoss()

        # Start training
        for _ in range(nb_epochs):
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                # Batch indexes
                begin, end = (
                    m * batch_size,
                    min((m + 1) * batch_size, x_preprocessed.shape[0]),
                )

                # Extract random batch
                i_batch = np.array(
                    [x_i for x_i in x_preprocessed[ind[begin: end]]] +
                    [np.array([0.1]), np.array([0.1, 0.2])]
                )[:-2]
                o_batch = y_preprocessed[ind[begin: end]]

                # Transform x into the model input space
                batch, batch_idx = self.transform_model_input(
                    x=i_batch, y=o_batch, compute_gradient=False)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                loss, sample_size_i, logging_output = self.task.train_step(
                    sample=batch,
                    model=self.model,
                    criterion=self.criterion,
                    optimizer=self._optimizer,
                )

    def transform_model_input(
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        y: Optional[np.ndarray] = None,
        compute_gradient: bool = False,
        tensor_input: bool = False,
        real_lengths: Optional[np.ndarray] = None,
    ):
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
        import torchaudio

        def _collate_fn(batch):
            # sort by seq length in descending order
            batch = sorted(batch, key=lambda t: t[0].size(0), reverse=True)
            batch_size = len(batch)
            max_seqlength = batch[0][0].size(0)
            feat_dim = batch[0][0].size(1)
            src_frames = torch.zeros(batch_size, max_seqlength, feat_dim)
            src_lengths = torch.zeros(batch_size, dtype=torch.int)

            # sort by target length in descending order (note: it won't change the order of "batch")
            batch_target = sorted(
                batch, key=lambda t: t[1].size(0), reverse=True)
            max_targetlength = batch_target[0][1].size(0)
            targets = torch.zeros(
                batch_size, max_targetlength, dtype=torch.long)
            for (sample, target) in batch:
                seq_length = sample.size(0)
                target_length = target.size(0)
                src_frames[i, :seq_length, :].copy_(feat)
                src_lengths[i] = seq_length
                targets[i, :target_length].copy_(target)

            batch_dict = {
                "net_input": {
                    "src_tokens": src_frames,
                    "src_lengths": src_lengths,
                },
                "target": targets,
            }
            return batch_dict

        # # These parameters are needed for the transformation
        # sample_rate = self._model.audio_conf.sample_rate
        # window_size = self._model.audio_conf.window_size
        # window_stride = self._model.audio_conf.window_stride

        # n_fft = int(sample_rate * window_size)
        # hop_length = int(sample_rate * window_stride)
        # win_length = n_fft

        # window = self._model.audio_conf.window.value

        # if window == "hamming":
        #     window_fn = torch.hamming_window
        # elif window == "hann":
        #     window_fn = torch.hann_window
        # elif window == "blackman":
        #     window_fn = torch.blackman_window
        # elif window == "bartlett":
        #     window_fn = torch.bartlett_window
        # else:
        #     raise NotImplementedError(
        #         "Spectrogram window %s not supported." % window)

        # # Create a transformer to transform between the two spaces
        # transformer = torchaudio.transforms.Spectrogram(
        #     n_fft=n_fft, hop_length=hop_length, win_length=win_length, window_fn=window_fn, power=None
        # )
        transformer = torchaudio.transforms.Spectrogram()
        transformer.to(self._device)

        # We must process each sequence separately due to the diversity of their length
        batch = []
        for i in range(len(x)):
            # First process the target
            if y is None:
                target = None
            else:
                sp = self.sp.EncoderAsPieces(y[i])
                target = self.dictionary.encode_line(
                    sp)  # target is a long tensor

            # Push the sequence to device
            if not tensor_input:
                x[i] = x[i].astype(ART_NUMPY_DTYPE)
                x[i] = torch.tensor(x[i]).to(self._device)

            # Set gradient computation permission
            if compute_gradient:
                x[i].requires_grad = True

            # Transform the sequence into the frequency space
            if tensor_input and real_lengths is not None:
                transformed_input = transformer(x[i][: real_lengths[i]])
            else:
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
        batch_idx = sorted(range(len(batch)),
                           key=lambda i: batch[i][0].size(1), reverse=True)

        # The collate function is important to convert input into model space
        batch_dict = _collate_fn(batch)

        return batch_dict, batch_idx

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

    def set_learning_phase(self, train: bool) -> None:
        raise NotImplementedError
