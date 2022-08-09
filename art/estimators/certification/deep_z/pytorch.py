# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2022
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
This module implements DeepZ proposed in Fast and Effective Robustness Certification.

| Paper link: https://papers.nips.cc/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf
"""

from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
import random

import warnings
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

from art.estimators.certification.deep_z.deep_z import ZonoConv, ZonoDenseLayer, ZonoReLU, ZonoBounds
from art.estimators.classification.pytorch import PyTorchClassifier
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor


class ConvertedModel(torch.nn.Module):
    """
    Class which converts the supplied pytorch model into an equivalent model
    which uses abstract operations
    """

    def __init__(self,
                 model: "torch.nn.Module",
                 channels_first: bool,
                 input_shape: Tuple[int, ...]):
        super().__init__()
        modules = []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # pylint: disable=W0613
        def forward_hook(input_module, hook_input, hook_output):
            modules.append(input_module)

        for module in model.children():
            module.register_forward_hook(forward_hook)

        if channels_first:
            input_for_hook = torch.rand(input_shape).to(device)
        else:
            raise ValueError("Please provide data in channels first format")

        input_for_hook = torch.unsqueeze(input_for_hook, dim=0)
        model(input_for_hook)  # hooks are fired sequentially from model input to the output

        self.ops = torch.nn.ModuleList()
        for module in modules:
            print("registered", type(module))
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                zono_conv = ZonoConv(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,  # type: ignore
                    stride=module.stride,  # type: ignore
                    dilation=module.dilation,  # type: ignore
                    padding=module.padding,  # type: ignore
                )
                zono_conv.conv.weight.data = module.weight.data.to(device)
                zono_conv.bias.data = module.bias.data.to(device)  # type: ignore
                self.ops.append(zono_conv)

            elif isinstance(module, torch.nn.modules.linear.Linear):
                zono_dense = ZonoDenseLayer(in_features=module.in_features, out_features=module.out_features)
                zono_dense.weight.data = module.weight.data.to(device)
                zono_dense.bias.data = module.bias.data.to(device)
                self.ops.append(zono_dense)

            elif isinstance(module, torch.nn.modules.activation.ReLU):
                self.ops.append(ZonoReLU(device=device))
            else:
                raise ValueError("Supported Operations are Conv2D, Linear, and RelU")

        for op_num, op in enumerate(self.ops):
            # as reshapes are not modules we infer when the reshape from convolutional to dense occurs
            if isinstance(op, ZonoDenseLayer):
                # if the preceeding op was a convolution:
                if isinstance(self.ops[op_num - 1], ZonoConv):
                    self.reshape_op_num = op_num
                    print("Inferred reshape on op num", op_num)
                # if the preceeding op was a relu and the one before the activation was a convolution
                if isinstance(self.ops[op_num - 1], ZonoReLU) and isinstance(self.ops[op_num - 2], ZonoConv):
                    self.reshape_op_num = op_num
                    print("Inferred reshape on op num", op_num)

    def forward(self, cent: np.ndarray, eps: np.ndarray) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Do the forward pass through the NN with the given error terms and zonotope center.

        :param eps: Error terms of the zonotope.
        :param cent: The datapoint, representing the zonotope center.
        :return: A tuple, the first element being the zonotope center vector.
                 The second is the zonotope error terms/coefficients.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        x = np.concatenate([cent, eps])
        x = torch.from_numpy(x.astype("float32")).to(device)

        for op_num, op in enumerate(self.ops):
            # as reshapes are not modules we infer when the reshape from convolutional to dense occurs
            if self.reshape_op_num == op_num:
                x = x.reshape((x.shape[0], -1))
            x = op(x)

        return x[0, :], x[1:, :]

    def concrete_forward(self, x: Union[np.ndarray, "torch.Tensor"]) -> "torch.Tensor":
        """
        Do the forward pass using the concrete operations

        :param x: regular (concrete) data.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype("float32")).to(device)

        for op_num, op in enumerate(self.ops):
            # as reshapes are not modules we infer when the reshape from convolutional to dense occurs
            if self.reshape_op_num == op_num:
                x = x.reshape((x.shape[0], -1))
            x = op.concrete_forward(x)
        return x


class PytorchDeepZ(PyTorchClassifier, ZonoBounds):
    """
    Implementation of DeepZ to certify neural network robustness. We use the zonotope representation of a datapoint as
    it travels through the network to then verify if it can have its class changed given a certain perturbation.

    | Paper link: https://papers.nips.cc/paper/2018/file/f2f446980d8e971ef3da97af089481c3-Paper.pdf
    """

    estimator_params = PyTorchClassifier.estimator_params

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
    ):
        """
        Create a certifier based on the zonotope domain.

        :param model: PyTorch model. The output of the model can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param optimizer: The optimizer used to train the classifier.
        :param channels_first: Set channels first or last.
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
        """

        warnings.warn(
            "\nThis estimator does not support networks which have dense layers before convolutional. "
            "We currently infer a reshape when a neural network goes from convolutional layers to "
            "dense layers. If your use case does not fall into this pattern then consider "
            "directly building a certifier network with the "
            "custom layers found in art.estimators.certification.deepz.deep_z.py\n"
        )
        converted_model = ConvertedModel(model, channels_first, input_shape)

        if optimizer is not None:
            import torch.optim as optim
            opt_state_dict = optimizer.state_dict()
            if TYPE_CHECKING:
                converted_optimizer: Union[optim.Adam, optim.SGD]

            if isinstance(optimizer, torch.optim.Adam):
                print("Converting Adam Optimiser")
                converted_optimizer = optim.Adam(converted_model.parameters(), lr=1e-4)
            elif isinstance(optimizer, torch.optim.SGD):
                print("Converting SGD Optimiser")
                converted_optimizer = optim.SGD(converted_model.parameters(), lr=1e-4)
            else:
                raise ValueError("Optimiser not supported for conversion")

            converted_optimizer.load_state_dict(opt_state_dict)

        super().__init__(
            model=converted_model,
            loss=loss,
            input_shape=input_shape,
            nb_classes=nb_classes,
            optimizer=converted_optimizer,
            channels_first=channels_first,
            clip_values=clip_values,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )

    def forward(self, cent: np.ndarray, eps: np.ndarray) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Do the forward pass through the NN with the given error terms and zonotope center.

        :param eps: Error terms of the zonotope.
        :param cent: The datapoint, representing the zonotope center.
        :return: A tuple, the first element being the zonotope center vector.
                 The second is the zonotope error terms/coefficients.
        """
        cent, eps = self.model.forward(cent, eps)
        return cent, eps

    def certify(self, cent: np.ndarray, eps: np.ndarray, prediction: int) -> bool:
        """
        Check if the datapoint has been certifiably classified.

        First do the forward pass through the NN with the given error terms and zonotope center to
        obtain the output zonotope.

        Then perform the certification step by computing the difference of the logits in the final zonotope
        and projecting to interval.

        :param eps: Error terms of the zonotope.
        :param cent: The datapoint, representing the zonotope center.
        :param prediction: The prediction the neural network gave on the basic datapoint.

        :return: True/False if the datapoint could be misclassified given the eps bounds.
        """
        cent_tensor, eps_tensor = self.forward(eps=eps, cent=cent)
        cent = cent_tensor.detach().cpu().numpy()
        eps = eps_tensor.detach().cpu().numpy()

        certification_results = []
        for k in range(self.nb_classes):
            if k != prediction:
                cert_via_sub = self.certify_via_subtraction(
                    predicted_class=prediction, class_to_consider=k, cent=cent, eps=eps
                )
                certification_results.append(cert_via_sub)

        return all(certification_results)

    @staticmethod
    def max_logit_loss(output, target):
        """
        Computes the loss as the largest logit value amongst the incorrect classes.
        """
        target_logit = output[:, target]
        output = output - target_logit

        ubs = torch.sum(torch.abs(output[1:, :]), axis=0) + output[0, :]

        loss = None
        for i in range(10):
            if i != target and (loss is None or ubs[i] > loss):
                loss = ubs[i]
        return loss

    @staticmethod
    def get_accuracy(preds: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
        """
        Helper function to print out the accuracy during training

        :param preds: (concrete) model predictions
        :param labels: ground truth labels (not one hot)

        :return: prediction accuracy
        """
        return torch.sum(torch.argmax(preds, dim=1) == labels) / len(labels)

    def make_adversarial_example(
        self,
        x: "torch.Tensor",
        y: "torch.Tensor",
        eps: float = 0.25,
        random_start: bool = True,
        num_steps: int = 20,
        step_size: float = 0.05,
    ) -> "torch.Tensor":
        clip_max = x + eps
        clip_min = x - eps

        clip_max = clip_max.cpu().numpy()
        clip_min = clip_min.cpu().numpy()

        if random_start:
            rand_start = torch.rand(size=x.shape).to(self.device) * 2 * eps
            rand_start = rand_start - eps
            x = x + rand_start
            x = torch.clamp(x, 0.0, 1.0)
            x = x.cpu().detach().numpy()
            x = np.clip(x, clip_min, clip_max)
            x = torch.from_numpy(x).to(self.device)

        for _ in range(num_steps):
            x.requires_grad = True
            # Forward pass the data through the model
            output = self.model.concrete_forward(x)

            # Calculate the loss
            loss = self._loss(output, y)

            # Zero all existing gradients
            self.model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = x.grad.data

            # Collect the element-wise sign of the data gradient
            sign_data_grad = data_grad.sign()

            # Create the perturbed image by adjusting each pixel of the input image
            x = x + step_size * sign_data_grad
            # Adding clipping to maintain [0,1] range
            x = torch.clamp(x, 0.0, 1.0)
            x = x.cpu().detach().numpy()
            x = np.clip(x, clip_min, clip_max)
            x = torch.from_numpy(x).to(self.device)
        return x

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        training_mode: bool = True,
        scheduler: Optional[Any] = None,
        bound: float = 0.25,
        certification_batch_size: int = 10,
        loss_weighting: float = 0.1,
        use_schedule: bool = True,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param pgd_batch_size: Size of batches to use for PGD training
        :param certification_batch_size: Size of batches to use for certified training. NB, this will run the data
                                         sequentially accumulating gradients over the batch size.
        :param loss_weighting:
        :param nb_epochs: Number of epochs to use for training.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """

        # Set model mode
        # self._model.train(mode=training_mode)
        pgd_batch_size = batch_size

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        num_batch = int(np.ceil(len(x_preprocessed) / float(pgd_batch_size)))
        ind = np.arange(len(x_preprocessed))
        from sklearn.utils import shuffle

        x_cert = np.copy(x_preprocessed)
        y_cert = np.copy(y_preprocessed)

        # Start training
        if use_schedule:
            step_per_epoch = bound / nb_epochs
            bound = 0.0

        for _ in tqdm(range(nb_epochs)):
            if use_schedule:
                bound += step_per_epoch
            # Shuffle the examples
            random.shuffle(ind)

            # Train for one epoch
            for m in range(num_batch):
                certified_loss = 0.0
                samples_certified = 0
                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # get the certified loss
                x_cert, y_cert = shuffle(np.array(x_cert), np.array(y_cert))
                for i, (sample, label) in enumerate(zip(x_cert, y_cert)):

                    eps_bound = np.eye(784) * bound
                    concrete_pred = self.model.concrete_forward(sample)
                    concrete_pred = torch.argmax(concrete_pred)
                    sample, eps_bound = self.pre_process(cent=sample, eps=eps_bound)
                    sample = np.expand_dims(sample, axis=0)

                    # Perform prediction
                    bias, eps = self.model(eps=eps_bound, cent=np.copy(sample))
                    # Form the loss function
                    bias = torch.unsqueeze(bias, dim=0)
                    certified_loss += self.max_logit_loss(
                        output=torch.cat((bias, eps)), target=np.expand_dims(label, axis=0)
                    )

                    certification_results = []
                    bias = torch.squeeze(bias).detach().cpu().numpy()
                    eps = eps.detach().cpu().numpy()

                    for k in range(self.nb_classes):
                        if k != concrete_pred:
                            cert_via_sub = self.certify_via_subtraction(
                                predicted_class=concrete_pred, class_to_consider=k, cent=bias, eps=eps
                            )
                            certification_results.append(cert_via_sub)

                    if all(certification_results):
                        samples_certified += 1

                    if (i - 1) % certification_batch_size == 0 and i > 0:
                        break

                certified_loss /= certification_batch_size

                # Concrete PGD loss
                i_batch = np.copy(x_preprocessed[ind[m * pgd_batch_size : (m + 1) * pgd_batch_size]])
                i_batch = torch.from_numpy(i_batch.astype("float32")).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * pgd_batch_size : (m + 1) * pgd_batch_size]]).to(
                    self._device
                )

                # Perform prediction
                i_batch = self.make_adversarial_example(i_batch, o_batch)
                self._optimizer.zero_grad()
                self.model.zero_grad()
                model_outputs = self.model.concrete_forward(i_batch)
                acc = self.get_accuracy(model_outputs, o_batch)

                # Form the loss function
                pgd_loss = self._loss(model_outputs, o_batch)
                print("Batch {}/{} Loss is {} Cert Loss is {}".format(m, num_batch, pgd_loss, certified_loss))
                print(
                    "Batch {}/{} Acc is {} Cert Acc is {}".format(
                        m, num_batch, acc, samples_certified / certification_batch_size
                    )
                )
                loss = certified_loss * loss_weighting + pgd_loss * (1 - loss_weighting)
                # Do training
                if self._use_amp:  # pragma: no cover
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss.backward()

                self._optimizer.step()
