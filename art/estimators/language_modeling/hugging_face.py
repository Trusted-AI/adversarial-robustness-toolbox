# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2023
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
This module implements the language model `HuggingFaceLanguage` using a PyTorch as a backend to interface with ART.
"""
import logging
from typing import List, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING

import numpy as np

from art.estimators.estimator import BaseEstimator
from art.estimators.language_modeling.language_model import LanguageModel

if TYPE_CHECKING:
    # pylint: disable=R0401
    import torch
    import transformers
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.defences.preprocessor.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class HuggingFaceLanguageModel(LanguageModel):
    """
    This class implements a language model with the Hugging Face framework and PyTorch backend.
    """

    estimator_params = BaseEstimator.estimator_params + [
        "device_type",
    ]

    def __init__(
        self,
        model: "transformers.PreTrainedModel",
        tokenizer: Union["transformers.PreTrainedTokenizerFast", "transformers.PreTrainedTokenizer"],
        loss: Optional["torch.nn.modules.loss._Loss"] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: Union["PREPROCESSING_TYPE", "Preprocessor"] = (0.0, 1.0),
        device_type: str = "gpu",
    ):
        """
        Initialization of HuggingFaceLanguageModel specifically for the PyTorch-based backend.

        :param model: Hugging Face model which returns outputs of type ModelOutput from the transformers library.
        :param tokenizer: Hugging Face tokenizer which converts str | list[str] into the model input.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
                     categorical, i.e. not converted to one-hot encoding. (Currently unused)
        :param optimizer: The optimizer used to train the classifier. (Currently unused)
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features. (Currently unused)
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator. (Currently unused)
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator. (Currently unused)
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input and the results will be
               divided by the second value. (Currently unused)
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        import torch

        super().__init__(
            model=model,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._model = model
        self._tokenizer = tokenizer
        self._loss = loss
        self._optimizer = optimizer
        self._device_type = device_type

        # Set device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device(f"cuda:{cuda_idx}")

        # Move model to GPU
        self._model.to(self._device)

    @property
    def device_type(self) -> str:
        """
        Return the type of device on which the estimator is run.

        :return: Type of device on which the estimator is run, either `gpu` or `cpu`.
        """
        return self._device_type

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    @property
    def model(self) -> "transformers.PreTrainedModel":
        """
        The model.

        :return: The model.
        """
        return self._model

    @property
    def tokenizer(self) -> Union["transformers.PreTrainedTokenizerFast", "transformers.PreTrainedTokenizer"]:
        """
        The tokenizer.

        :return: The tokenizer.
        """
        return self._tokenizer

    @property
    def loss(self) -> "torch.nn.modules.loss._Loss":
        """
        Return the loss function.

        :return: The loss function.
        """
        return self._loss  # type: ignore

    @property
    def optimizer(self) -> "torch.optim.Optimizer":
        """
        Return the optimizer.

        :return: The optimizer.
        """
        return self._optimizer  # type: ignore

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        raise NotImplementedError

    def tokenize(self, x: Union[str, List[str]], **kwargs) -> Dict[str, list]:
        """
        Use the tokenizer to encode a string to a list of token ids or lists of strings to a lists of token ids.

        :param x: A string or list of strings.
        :param kwargs: Additional keyword arguments for the tokenizer. Can override the `x` input.
        :return: A dictionary of the tokenized string or multiple strings. The fields of the dictionary will be
                 specific to the tokenizer and keyword arguments provided.
        """
        # Create input parameters
        inputs = {"text": x}
        for key, value in kwargs.items():
            inputs[key] = value

        tokenized_output = self._tokenizer(**inputs)
        tokenized_dict = tokenized_output.data
        return tokenized_dict

    def encode(self, x: str, **kwargs) -> List[int]:
        """
        Use the tokenizer to encode a string to a list of token ids.

        :param x: A string to convert to tokens.
        :param kwargs: Additional keyword arguments for the tokenizer. Can override the `x` input.
        :return: A list of encoded token ids of a string.
        """
        # Create input parameters
        inputs = {"text": x}
        for key, value in kwargs.items():
            inputs[key] = value

        encoded_output = self._tokenizer.encode(**inputs)
        return encoded_output

    def batch_encode(self, x: List[str], **kwargs) -> List[List[int]]:
        """
        Use the tokenizer to encode a list of strings to lists of token ids.

        :param x: A list strings to convert to tokens.
        :param kwargs: Additional keyword arguments for the tokenizer. Can override the `x` input.
        :return: lists of encoded token ids of each string.
        """
        return [self.encode(x_i, **kwargs) for x_i in x]

    def decode(self, x: List[int], **kwargs) -> str:
        """
        Use the tokenizer to decode a list of token ids to a string.

        :param x: A list of tokenized input ids of a string.
        :param kwargs: Additional keyword arguments for the tokenizer. Can override the `x` input.
        :return: Decoded string.
        """
        # Create input parameters
        inputs = {"token_ids": x}
        for key, value in kwargs.items():
            inputs[key] = value

        # Decode tokens to string
        decoded_output = self._tokenizer.decode(**inputs)
        return decoded_output

    def batch_decode(self, x: List[List[int]], **kwargs) -> List[str]:
        """
        Use the tokenizer to decode lists of token ids to a list of strings.

        :param x: List of tokenized input ids of multiple strings.
        :param kwargs: Additional keyword arguments for the tokenizer. Can override the `x` input.
        :return: List of decoded strings.
        """
        # Create input parameters
        inputs = {"sequences": x}
        for key, value in kwargs.items():
            inputs[key] = value

        # Decode tokens to string
        decoded_output = self._tokenizer.batch_decode(**inputs)
        return decoded_output

    def predict(self, x: Optional[Union[str, List[str]]] = None, **kwargs) -> Dict[str, np.ndarray]:
        """
        Tokenize the string or list of strings and run inference on the model.

        :param text: A string or list of strings.
        :param kwargs: Additional keyword arguments for the model. Can override the `x` input.
        :return: A dictionary of the model output from running inference on the input string or strings. The fields
                 of the dictionary will be specific to the model used.
        """
        import torch

        self._model.eval()

        inputs = {}

        # Tokenize text input
        if x is not None:
            tokenized = self._tokenizer(x, padding=True, truncation=True, return_tensors="pt")
            for key, value in tokenized.items():
                inputs[key] = value.to(self._device)

        # Convert inputs to tensors and move to GPU
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self._device)
            elif isinstance(value, np.ndarray):
                inputs[key] = torch.from_numpy(value).to(self._device)
            elif isinstance(value, list):
                if isinstance(value[0], torch.Tensor):
                    inputs[key] = [v_i.to(self._device) for v_i in value]
                elif isinstance(value[0], np.ndarray):
                    inputs[key] = [torch.from_numpy(v_i).to(self._device) for v_i in value]
                elif isinstance(value[0], (float, int)):
                    inputs[key] = torch.tensor(value).to(self._device)
                else:
                    inputs[key] = value
            else:
                inputs[key] = value

        # Run prediction
        with torch.no_grad():
            out = self._model(**inputs)

        # Extract outputs and convert to numpy arrays
        outputs = {}
        for key, value in out.items():
            if isinstance(value, torch.Tensor):
                outputs[key] = value.detach().cpu().numpy()
            elif isinstance(value, (tuple, list)):
                if isinstance(value[0], torch.Tensor):
                    outputs[key] = [v_i.detach().cpu().numpy() for v_i in value]
                elif isinstance(value[0], (tuple, list)):
                    outputs[key] = [[v_ij.detach().cpu().numpy() for v_ij in v_i] for v_i in value]
                else:
                    outputs[key] = value
            else:
                outputs[key] = value

        return outputs

    def generate(self, x: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """
        Tokenize the string or list of strings and run generation using the model.

        :param text: A string or list of strings.
        :param kwargs: Additional keyword arguments for the model. Can override the `x` input.
        :return: A dictionary of the model output from running inference on the input string or strings. The fields
                 of the dictionary will be specific to the model used.
        """
        import torch

        self._model.eval()

        inputs = {}

        # Tokenize text input
        if x is not None:
            tokenized = self._tokenizer(x, padding=True, truncation=True, return_tensors="pt")
            for key, value in tokenized.items():
                inputs[key] = value.to(self._device)

        # Convert inputs to tensors and move to GPU
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self._device)
            elif isinstance(value, np.ndarray):
                inputs[key] = torch.from_numpy(value).to(self._device)
            elif isinstance(value, list):
                if isinstance(value[0], torch.Tensor):
                    inputs[key] = [v_i.to(self._device) for v_i in value]
                elif isinstance(value[0], np.ndarray):
                    inputs[key] = [torch.from_numpy(v_i).to(self._device) for v_i in value]
                elif isinstance(value[0], (float, int)):
                    inputs[key] = torch.tensor(value).to(self._device)
                else:
                    inputs[key] = value
            else:
                inputs[key] = value
        inputs["return_dict_in_generate"] = False

        # Run prediction
        with torch.no_grad():
            tokens = self._model.generate(**inputs)

        # Decode output tokens
        generated_strings = self._tokenizer.batch_decode(tokens, skip_special_tokens=True)

        if isinstance(x, str):
            return generated_strings[0]

        return generated_strings

    def fit(self, x: Any, y: Any, **kwargs):
        """
        Fit the estimator using the training data `(x, y)`.

        :param x: Training data.
        :type x: Format as expected by the `model`
        :param y: Target values.
        :type y: Format as expected by the `model`
        """
        raise NotImplementedError

    def _set_layer(self, train: bool, layerinfo: List["torch.nn.modules.Module"]) -> None:
        """
        Set all layers that are an instance of `layerinfo` into training or evaluation mode.

        :param train: False for evaluation mode.
        :param layerinfo: List of module types.
        """
        import torch

        assert all((issubclass(layer, torch.nn.modules.Module) for layer in layerinfo))  # type: ignore

        def set_train(layer, layerinfo=layerinfo):
            "Set layer into training mode if instance of `layerinfo`."
            if isinstance(layer, tuple(layerinfo)):
                layer.train()

        def set_eval(layer, layerinfo=layerinfo):
            "Set layer into evaluation mode if instance of `layerinfo`."
            if isinstance(layer, tuple(layerinfo)):
                layer.eval()

        if train:
            self._model.apply(set_train)
        else:
            self._model.apply(set_eval)

    def set_dropout(self, train: bool) -> None:
        """
        Set all dropout layers into train or eval mode.

        :param train: False for evaluation mode.
        """
        import torch

        # pylint: disable=W0212
        self._set_layer(train=train, layerinfo=[torch.nn.modules.dropout._DropoutNd])  # type: ignore

    def set_batchnorm(self, train: bool) -> None:
        """
        Set all batch normalization layers into train or eval mode.

        :param train: False for evaluation mode.
        """
        import torch

        # pylint: disable=W0212
        self._set_layer(train=train, layerinfo=[torch.nn.modules.batchnorm._BatchNorm])  # type: ignore

    def set_multihead_attention(self, train: bool) -> None:
        """
        Set all multi-head attention layers into train or eval mode.

        :param train: False for evaluation mode.
        """
        import torch

        # pylint: disable=W0212
        self._set_layer(train=train, layerinfo=[torch.nn.modules.MultiheadAttention])  # type: ignore
