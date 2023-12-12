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
This module implements the abstract estimator `HuggingFaceClassifier` using the PyTorchClassifier as a backend
to interface with ART.
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
    This class implements a classifier with the HuggingFace framework.
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
    
    def tokenize(self, x: Union[str, List[str]], **kwargs) -> Dict[str, Any]:
        tokenized_output = self._tokenizer(x, **kwargs)
        tokenized_dict = tokenized_output.data
        return tokenized_dict
    
    def encode(self, x: Union[str, List[str]], **kwargs) -> np.ndarray:
        encoded_output = self._tokenizer.encode(x, **kwargs)
        return encoded_output
    
    def decode(
        self,
        token_ids: np.ndarray,
        *,
        skip_special_tokens: Optional[bool] = None,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs
    ) -> str:
        """
        Decode token ids to a string using the tokenizer.

        :param token_ids: The tokenized input ids of a string.
        :param skip_special_tokens: Whether or not to remove special tokens in the decoding.
        :param clean_up_tokenization_spaces: Whether or not to clean up the tokenization spaces.
        """
        # Create input parameters
        inputs = {}
        inputs['token_ids'] = token_ids
        if skip_special_tokens:
            inputs['skip_special_tokens'] = skip_special_tokens
        if clean_up_tokenization_spaces:
            inputs['skip_special_tokens'] = clean_up_tokenization_spaces

        decoded_output = self._tokenizer.decode(**inputs, **kwargs)
        return decoded_output
    
    def predict(self, x: Optional[Union[str, List[str]]] = None, **kwargs) -> Dict[str, Any]:
        import torch

        self._model.eval()
        
        inputs = {}

        # Tokenize text input
        if x is not None:
            tokenized = self._tokenizer(x, padding=True, truncation=True, return_tensors="pt")
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
    
    def generate(self, x: Any, **kwargs) -> Any:
        raise NotImplementedError

    def fit(self, x: Any, **kwargs):
        raise NotImplementedError

