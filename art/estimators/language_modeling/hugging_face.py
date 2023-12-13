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

from typing import List, Optional, Union, Dict, Any, TYPE_CHECKING

import numpy as np

from art.estimators.language_modeling.language_model import LanguageModel

if TYPE_CHECKING:
    import torch
    import transformers

logger = logging.getLogger(__name__)


class HuggingFaceLanguageModel(LanguageModel):
    """
    This class implements a language model with the Hugging Face framework and PyTorch backend.
    """
    
    def __init__(
        self,
        model: "transformers.PreTrainedModel",
        tokenizer: Union["transformers.PreTrainedTokenizerFast", "transformers.PreTrainedTokenizer"],
        loss: Optional["torch.nn.modules.loss._Loss"] = None,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        device_type: str = "gpu",
    ):
        """
        Initialization of HuggingFaceLanguageModel specifically for the PyTorch-based backend.

        :param model: Hugging Face model which returns outputs of type ModelOutput from the transformers library.
        :param tokenizer: Hugging Face tokenizer which converts str | list[str] into the model input.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
                     categorical, i.e. not converted to one-hot encoding.
        :param optimizer: The optimizer used to train the classifier.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        """
        import torch

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            loss=loss,
            optimizer=optimizer,
            device_type=device_type,
        )

        self._model = model
        self._tokenizer = tokenizer
        self._loss = loss
        self._optimizer = optimizer

        # Move model to GPU
        self._model.to(self._device)

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    @property
    def model(self) -> "transformers.PreTrainedModel":
        return self._model

    @property
    def tokenizer(self) -> Union["transformers.PreTrainedTokenizerFast", "transformers.PreTrainedTokenizer"]:
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
    
    def tokenize(self, text: Union[str, List[str]], **kwargs) -> Dict[str, np.ndarray]:
        """
        Use the tokenizer to encode a string to a list of token ids or lists of strings to a lists of token ids.

        :param text: A string or list of strings.
        :param kwargs: Additional keyword arguments for the tokenizer. Can override the `text` input.
        :return: A dictionary of the tokenized string or multiple strings.
        """
        # Create input parameters
        inputs = {'text': text}
        for k, v in kwargs.items():
            inputs[k] = v
        inputs['return_tensors'] = 'np'

        tokenized_output = self._tokenizer(**inputs)
        tokenized_dict = tokenized_output.data
        return tokenized_dict
    
    def encode(self, text: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Use the tokenizer to encode a string to a list of token ids or lists of strings to a lists of token ids.

        :param text: A string or list of strings.
        :param kwargs: Additional keyword arguments for the tokenizer. Can override the `text` input.
        :return: A list of encoded token ids of a string or lists of encoded token ids of multiple strings.
        """
        # Encode each string individually
        if isinstance(text, list):
            return [self.encode(t, **kwargs) for t in text]

        # Encode single string
        encoded_output = self._tokenizer.encode(text, **kwargs)

        return np.asarray(encoded_output)
    
    def decode(
        self,
        tokens: Union[np.ndarray, List[np.ndarray]],
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Use the tokenizer to decode a list of token ids to a string or lists of token ids to a list of strings.

        :param tokens: A list of tokenized input ids of a string or lists of tokenized input ids of strings.
        :param kwargs: Additional keyword arguments for the tokenizer. Can override the `tokens` input.
        :return: Decoded string or list of strings.
        """
        if isinstance(tokens, list):
            sequences = isinstance(tokens[0], (np.ndarray, list))
        else:
            sequences = len(tokens.shape) > 1

        # Create input parameters
        inputs = {}
        if sequences:
            inputs['sequences'] = tokens
        else:
            inputs['token_ids'] = tokens
        for k, v in kwargs.items():
            inputs[k] = v
        
        # Decode tokens to string
        if sequences:
            decoded_output = self._tokenizer.batch_decode(**inputs)
        else:
            decoded_output = self._tokenizer.decode(**inputs)
        return decoded_output
    
    def predict(self, text: Optional[Union[str, List[str]]] = None, **kwargs) -> Dict[str, np.ndarray]:
        """
        Tokenize the string or list of strings and run inference on the model.

        :param text: A string or list of strings.
        :param kwargs: Additional keyword arguments for the model. Can override the `text` input.
        :return: A dictionary of the model output from running inference on the input string or strings.
        """
        import torch

        self._model.eval()
        
        inputs = {}

        # Tokenize text input
        if text is not None:
            tokenized = self._tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            for k, v in tokenized.items():
                inputs[k] = v.to(self._device)

        # Convert inputs to tensors and move to GPU
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self._device)
            elif isinstance(v, np.ndarray):
                inputs[k] = torch.from_numpy(v).to(self._device)
            elif isinstance(v, list):
                if isinstance(v[0], torch.Tensor):
                    inputs[k] = [v_i.to(self._device) for v_i in v]
                elif isinstance(v[0], np.ndarray):
                    inputs[k] = [torch.from_numpy(v_i).to(self._device) for v_i in v]
                elif isinstance(v[0], (float, int)):
                    inputs[k] = torch.tensor(v).to(self._device)
                else:
                    inputs[k] = v
            else:
                inputs[k] = v

        # Run prediction
        with torch.no_grad():
            out = self._model(**inputs)
        
        # Extract outputs and convert to numpy arrays
        outputs = {}
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                outputs[k] = v.detach().cpu().numpy()
            elif isinstance(v, (tuple, list)):
                if isinstance(v[0], torch.Tensor):
                    outputs[k] = [v_i.detach().cpu().numpy() for v_i in v]
                elif isinstance(v[0], (tuple, list)):
                    outputs[k] = [[v_ij.detach().cpu().numpy() for v_ij in v_i] for v_i in v]
                else:
                    outputs[k] = v
            else:
                outputs[k] = v

        return outputs
    
    def generate(self, text: Any, **kwargs) -> Any:
        raise NotImplementedError

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

