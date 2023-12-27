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

from typing import List, Optional, Tuple, Union, Dict, Callable, Any, TYPE_CHECKING

import numpy as np
import six

from art.estimators.classification.pytorch import PyTorchClassifier

if TYPE_CHECKING:
    import torch
    import transformers
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
    from transformers.modeling_outputs import ImageClassifierOutput

logger = logging.getLogger(__name__)


class HuggingFaceClassifierPyTorch(PyTorchClassifier):
    """
    This class implements a classifier with the HuggingFace framework.
    """

    def __init__(
        self,
        model: "transformers.PreTrainedModel",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        use_amp: bool = False,
        opt_level: str = "O1",
        loss_scale: Optional[Union[float, str]] = "dynamic",
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        processor: Optional[Callable] = None,
        device_type: str = "gpu",
    ):
        """
        Initialization of HuggingFaceClassifierPyTorch specifically for the PyTorch-based backend.

        :param model: Huggingface model model which returns outputs of type
                      ImageClassifierOutput from the transformers library.
                      Must have the logits attribute set as output.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
                categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param optimizer: The optimizer used to train the classifier.
        :param use_amp: Whether to use the automatic mixed precision tool to enable mixed precision training or
                        gradient computation, e.g. with loss gradient computation. When set to True, this option is
                        only triggered if there are GPUs available.
        :param opt_level: Specify a pure or mixed precision optimization level. Used when use_amp is True. Accepted
                            values are `O0`, `O1`, `O2`, and `O3`.
        :param loss_scale: Loss scaling. Used when use_amp is True. If passed as a string, must be a string
                            representing a number, e.g., “1.0”, or the string “dynamic”.
        :param nb_classes: The number of classes of the model.
        :param optimizer: The optimizer used to train the classifier.
        :param channels_first: Set channels first or last. Normally should be set to True for HF models based on
                               a pytorch backend.
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
                maximum values allowed for features. If floats are provided, these will be used as the range of all
                features. If arrays are provided, each value will be considered the bound for a feature, thus
                the shape of clip values needs to match the total number of features.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
                used for data preprocessing. The first value will be subtracted from the input. The input will then
                be divided by the second one.
        :param device_type: Type of device on which the classifier is run, either `gpu` or `cpu`.
        :param processor: Optional argument. Function which takes in a batch of data and performs
                          the preprocessing relevant to a given foundation model.
                          Must be differentiable for grandient based defences and attacks.
        """
        import torch

        self.processor = processor

        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            use_amp=use_amp,
            opt_level=opt_level,
            loss_scale=loss_scale,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )

        import functools

        def prefix_function(function: Callable, postfunction: Callable) -> Callable[[Any, Any], torch.Tensor]:
            """
            Huggingface returns logit under outputs.logits. To make this compatible with ART we wrap the forward pass
            function of a HF model here, which automatically extracts the logits.

            :param function: The first function to run, in our case the forward pass of the model.
            :param postfunction: Second function to run, in this case simply extracts the logits.
            :return: model outputs.
            """

            @functools.wraps(function)
            def run(*args, **kwargs) -> torch.Tensor:
                outputs = function(*args, **kwargs)
                return postfunction(outputs)

            return run

        def get_logits(outputs: "ImageClassifierOutput") -> torch.Tensor:
            """
            Gets the logits attribute from ImageClassifierOutput

            :param outputs: outputs of type ImageClassifierOutput from a Huggingface
            :return: model's logit predictions.
            """
            if isinstance(outputs, torch.Tensor):
                return outputs
            return outputs.logits

        self.model.forward = prefix_function(self.model.forward, get_logits)  # type: ignore

    def _make_model_wrapper(self, model: "torch.nn.Module") -> "torch.nn.Module":
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        import torch

        input_shape = self._input_shape
        input_for_hook = torch.rand(input_shape)
        # self.device may not match the device the raw model was passed into ART.
        # Check if the model is on cuda, if so set the hook input accordingly
        if next(model.parameters()).is_cuda:
            cuda_idx = torch.cuda.current_device()
            input_for_hook = input_for_hook.to(torch.device(f"cuda:{cuda_idx}"))

        input_for_hook = torch.unsqueeze(input_for_hook, dim=0)

        if self.processor is not None:
            input_for_hook = self.processor(input_for_hook)

        processor = self.processor
        try:
            # Define model wrapping class only if not defined before
            if not hasattr(self, "_model_wrapper"):

                class ModelWrapper(torch.nn.Module):
                    """
                    This is a wrapper for the input model.
                    """

                    def __init__(self, model: torch.nn.Module):
                        """
                        Initialization by storing the input model.

                        :param model: PyTorch model. The forward function of the model must return the logit output.
                        """
                        super().__init__()
                        self._model = model

                    # pylint: disable=W0221
                    # disable pylint because of API requirements for function
                    def forward(self, x):
                        """
                        This is where we get outputs from the input model.

                        :param x: Input data.
                        :type x: `torch.Tensor`
                        :return: a list of output layers, where the last 2 layers are logit and final outputs.
                        :rtype: `list`
                        """
                        # pylint: disable=W0212
                        # disable pylint because access to _model required

                        result = []

                        if isinstance(self._model, torch.nn.Module):
                            if processor is not None:
                                x = processor(x)
                            x = self._model.forward(x)
                            result.append(x)

                        else:  # pragma: no cover
                            raise TypeError("The input model must inherit from `nn.Module`.")

                        return result

                    @property
                    def get_layers(self) -> List[str]:
                        """
                        Return the hidden layers in the model, if applicable.

                        :return: The hidden layers in the model, input and output layers excluded.

                        .. warning:: `get_layers` tries to infer the internal structure of the model.
                                     This feature comes with no guarantees on the correctness of the result.
                                     The intended order of the layers tries to match their order in the model, but this
                                     is not guaranteed either.
                        """

                        result_dict = {}

                        modules = []

                        # pylint: disable=W0613
                        def forward_hook(input_module, hook_input, hook_output):
                            logger.info("input_module is %s with id %i", input_module, id(input_module))
                            modules.append(id(input_module))

                        handles = []

                        for name, module in self._model.named_modules():
                            logger.info(
                                "found %s with type %s and id %i and name %s with submods %i ",
                                module,
                                type(module),
                                id(module),
                                name,
                                len(list(module.named_modules())),
                            )

                            if name != "" and len(list(module.named_modules())) == 1:
                                handles.append(module.register_forward_hook(forward_hook))
                                result_dict[id(module)] = name

                        logger.info("mapping from id to name is %s", result_dict)

                        logger.info("------ Finished Registering Hooks------")
                        model(input_for_hook)  # hooks are fired sequentially from model input to the output

                        logger.info("------ Finished Fire Hooks------")

                        # Remove the hooks
                        for hook in handles:
                            hook.remove()

                        logger.info("new result is: ")
                        name_order = []
                        for module in modules:
                            name_order.append(result_dict[module])

                        logger.info(name_order)

                        return name_order

                # Set newly created class as private attribute
                self._model_wrapper = ModelWrapper  # type: ignore

            # Use model wrapping class to wrap the PyTorch model received as argument
            return self._model_wrapper(model)

        except ImportError:  # pragma: no cover
            raise ImportError("Could not find PyTorch (`torch`) installation.") from ImportError

    def get_activations(  # type: ignore
        self,
        x: Union[np.ndarray, "torch.Tensor"],
        layer: Optional[Union[int, str]] = None,
        batch_size: int = 128,
        framework: bool = False,
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and
        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by
        calling `layer_names`.

        :param x: Input for computing the activations.
        :param layer: Layer for computing the activations
        :param batch_size: Size of batches.
        :param framework: If true, return the intermediate tensor representation of the activation.
        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.
        """
        import torch

        self._model.eval()

        # Apply defences
        if framework:
            no_grad = False
        else:
            no_grad = True
        x_preprocessed, _ = self._apply_preprocessing(x=x, y=None, fit=False, no_grad=no_grad)

        # Get index of the extracted layer
        if isinstance(layer, six.string_types):
            if layer not in self._layer_names:  # pragma: no cover
                raise ValueError(f"Layer name {layer} not supported")
            layer_index = self._layer_names.index(layer)

        elif isinstance(layer, int):
            layer_index = layer

        else:  # pragma: no cover
            raise TypeError("Layer must be of type str or int")

        def get_feature(name):
            # the hook signature
            def hook(model, input, output):  # pylint: disable=W0622,W0613
                # TODO: this is using the input, rather than the output, to circumvent the fact
                # TODO: that flatten is not a layer in pytorch, and the activation defence expects
                # TODO: a flattened input. A better option is to refactor the activation defence
                # TODO: to not crash if non 2D inputs are provided.
                self._features[name] = input

            return hook

        if not hasattr(self, "_features"):
            self._features: Dict[str, torch.Tensor] = {}
            # register forward hooks on the layers of choice
        handles = []

        lname = self._layer_names[layer_index]

        if layer not in self._features:
            for name, module in self.model.named_modules():
                if name == lname and len(list(module.named_modules())) == 1:
                    handles.append(module.register_forward_hook(get_feature(name)))

        if framework:
            if isinstance(x_preprocessed, torch.Tensor):
                self._model(x_preprocessed)
                return self._features[self._layer_names[layer_index]][0]
            input_tensor = torch.from_numpy(x_preprocessed)
            self._model(input_tensor.to(self._device))
            return self._features[self._layer_names[layer_index]][0]  # pylint: disable=W0212

        # Run prediction with batch processing
        results = []
        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))

        for m in range(num_batch):
            # Batch indexes
            begin, end = (
                m * batch_size,
                min((m + 1) * batch_size, x_preprocessed.shape[0]),
            )

            # Run prediction for the current batch
            self._model(torch.from_numpy(x_preprocessed[begin:end]).to(self._device))
            layer_output = self._features[self._layer_names[layer_index]]  # pylint: disable=W0212

            if isinstance(layer_output, tuple):
                results.append(layer_output[0].detach().cpu().numpy())
            else:
                results.append(layer_output.detach().cpu().numpy())

        results_array = np.concatenate(results)
        return results_array
