import torch
import numpy as np
import random
import logging
import six
from tqdm import tqdm

from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
from art.estimators.classification.pytorch import PyTorchClassifier
from art.utils import check_and_transform_label_format
logger = logging.getLogger(__name__)


class HuggingFaceClassifier(PyTorchClassifier):

    def __init__(self,
                 model,
                 loss: "torch.nn.modules.loss._Loss",
                 input_shape: Tuple[int, ...],
                 nb_classes: int,
                 optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
                 clip_values: Optional["CLIP_VALUES_TYPE"] = None,
                 preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0), processor=None):
        import transformers

        assert isinstance(model, transformers.PreTrainedModel)

        self.processor = processor

        super().__init__(
            model=model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=optimizer,
            channels_first=True,
            clip_values=clip_values,
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=preprocessing,
            device_type='gpu')

        import functools

        def prefix_function(function, postfunction):
            """
            Huggingface returns logit under outputs.logits.
            To make this compatible with ART we wrap the forward pass function
            of a HF model here, which automatically extracts the logits.
            """
            @functools.wraps(function)
            def run(*args, **kwargs):
                outputs = function(*args, **kwargs)
                return postfunction(outputs)
            return run

        def get_logits(outputs):
            if isinstance(outputs, torch.Tensor):
                return outputs
            return outputs.logits

        self.model.forward = prefix_function(self.model.forward, get_logits)

    def __call__(self, image):

        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image.to(self._device)
            outputs = self.model(**image)
        else:
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).to(self._device)
            outputs = self.model(image)
        return outputs

    def forward(self, image):
        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image.to(self._device)
            outputs = self.model(**image)
        else:
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).to(self._device)
            outputs = self.model(image)
        return outputs

    def _make_model_wrapper(self, model: "torch.nn.Module") -> "torch.nn.Module":
        # Try to import PyTorch and create an internal class that acts like a model wrapper extending torch.nn.Module
        input_shape = self._input_shape
        try:
            import torch

            # Define model wrapping class only if not defined before
            if not hasattr(self, "_model_wrapper"):

                class ModelWrapper(torch.nn.Module):
                    """
                    This is a wrapper for the input model.
                    """

                    import torch

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
                        if isinstance(self._model, torch.nn.Sequential):
                            for _, module_ in self._model._modules.items():
                                x = module_(x)
                                result.append(x)

                        elif isinstance(self._model, torch.nn.Module):
                            x = self._model(x)
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
                                     is not guaranteed either. In addition, the function can only infer the internal
                                     layers if the input model is of type `nn.Sequential`, otherwise, it will only
                                     return the logit layer.
                        """
                        import torch
                        import transformers
                        result_dict = {}

                        modules = []

                        def forward_hook(input_module, hook_input, hook_output):
                            print(f'input_module is {input_module} with id {id(input_module)}')
                            modules.append(id(input_module))

                        handles = []

                        for name, module in self._model.named_modules():
                            print(f'found {module} with type {type(module)} and id {id(module)} '
                                  f'and name {name} with submods {len(list(module.named_modules()))}')
                            if name != '' and len(list(module.named_modules())) == 1:
                                handles.append(module.register_forward_hook(forward_hook))
                                result_dict[id(module)] = name

                        print('\n')
                        print('mapping from id to name is ', result_dict)

                        print('------ Finished Registering Hooks------')
                        input_for_hook = torch.rand(input_shape)
                        print(input_for_hook.shape)
                        input_for_hook = torch.unsqueeze(input_for_hook, dim=0)
                        model(input_for_hook)  # hooks are fired sequentially from model input to the output

                        print('------ Finished Fire Hooks------')

                        # Remove the hooks
                        for h in handles:
                            h.remove()

                        print('new result is ')
                        name_order = []
                        for module in modules:
                            name_order.append(result_dict[module])

                        print(name_order)

                        return name_order

                # Set newly created class as private attribute
                self._model_wrapper = ModelWrapper

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
                # TODO: this is using the input, rather than the output, to circumvent the fact that flatten is not a layer
                # TODO: in pytorch, and the activation defence expects a flattened input. A better option is to
                # TODO: refactor the activation defence to not crash if non 2D inputs are provided.
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

            if isinstance(layer_output, Tuple):
                results.append(layer_output[0].detach().cpu().numpy())
            else:
                results.append(layer_output.detach().cpu().numpy())

        results_array = np.concatenate(results)
        return results_array

    def get_grad(self, image, labels, loss_fn):
        """
        Get gradient wrt input image.
        Testing function. To be removed in final PR

        :param image:
        :param labels:
        :return:
        """

        if not isinstance(image, torch.Tensor):
            labels = torch.from_numpy(labels)
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        image.requires_grad = True
        self.model.eval()
        self.model.zero_grad()

        if self.processor is not None:
            image = self.processor(images=image, return_tensors="pt")
            image.to(self._device)
            image['pixel_values'].requires_grad = True
            loss = self.model(**image, labels=labels)[0]
        else:
            out = self.model(image)
            loss = loss_fn(out, labels)
        loss.backward()
        self.model.eval()

        return image.grad

    def make_adv_example(self, x, y):
        """
        Testing function: to be removed in final PR
        """
        self.epsilon = 8 / 255
        self.attack_lr = 1 / 255
        upsampler = torch.nn.Upsample(scale_factor=7, mode='nearest')

        x = x.to(self._device)
        y = y.to(self._device)

        x = upsampler(x)  # hard code resize for now
        model_outputs = self.model(x)
        acc = self.get_accuracy(model_outputs, y)
        print('clean acc is ', acc)

        x_adv = x.detach().clone()
        x_adv.requires_grad = True

        for _ in range(30):
            self.model.zero_grad()
            # x_adv.zero_grad()
            grad = self.get_grad(x_adv, y, loss_fn=torch.nn.CrossEntropyLoss())
            with torch.no_grad():
                grad = grad.sign()
                x_adv = x_adv + self.attack_lr * grad

                # Projection
                noise = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(x + noise, min=0, max=1)

        model_outputs = self.model(x_adv)
        acc = self.get_accuracy(model_outputs, y)
        print('adv acc is ', acc)

    def train(self, x, y,
              batch_size: int = 128,
              nb_epochs: int = 10,
              training_mode: bool = True,
              drop_last: bool = False,
              scheduler: Optional[Any] = None,
              verbose=True,
              **kwargs,):
        import torch

        # Set model mode
        self.model.train()

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        num_batch = len(x_preprocessed) / float(batch_size)
        if drop_last:
            num_batch = int(np.floor(num_batch))
        else:
            num_batch = int(np.ceil(num_batch))
        ind = np.arange(len(x_preprocessed))

        # Start training
        for _ in tqdm(range(nb_epochs)):
            # Shuffle the examples
            random.shuffle(ind)
            pbar = tqdm(range(num_batch), disable=not verbose)

            epoch_loss = []
            epoch_acc = []

            # Train for one epoch
            for m in pbar:
                i_batch = np.copy(x_preprocessed[ind[m * batch_size: (m + 1) * batch_size]])
                i_batch = torch.from_numpy(i_batch).to(self._device)
                # i_batch = upsampler(i_batch)  # hard code resize for now
                i_batch = self.processor(i_batch)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                model_outputs = self.model(i_batch)
                acc = self.get_accuracy(model_outputs.logits, o_batch)

                # Form the loss function
                loss = self._loss(model_outputs.logits, o_batch)

                # Do training
                if self._use_amp:  # pragma: no cover
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss.backward()

                self._optimizer.step()
                epoch_loss.append(loss)
                epoch_acc.append(acc)

                if verbose:
                    pbar.set_description(
                        f"Loss {torch.mean(torch.stack(epoch_loss)):.2f} "
                        f"Acc {np.mean(epoch_acc):.2f}"
                    )

            if scheduler is not None:
                scheduler.step()

            torch.save(self.model.state_dict(), 'hf_model.pt')

    @staticmethod
    def get_accuracy(preds: Union[np.ndarray, "torch.Tensor"], labels: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Helper function to print out the accuracy during training

        :param preds: model predictions
        :param labels: ground truth labels (not one hot)
        :return: prediction accuracy
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        return np.sum(np.argmax(preds, axis=1) == labels) / len(labels)
