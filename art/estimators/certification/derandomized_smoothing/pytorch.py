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
This module implements De-Randomized smoothing approaches PyTorch.

(De)Randomized Smoothing for Certifiable Defense against Patch Attacks

| Paper link: https://arxiv.org/abs/2002.10733

and

Certified Patch Robustness via Smoothed Vision Transformers

| Paper link Accepted version:
    https://openaccess.thecvf.com/content/CVPR2022/papers/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.pdf

| Paper link Arxiv version (more detail): https://arxiv.org/pdf/2110.07719.pdf
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import importlib
import logging
from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
import random

import numpy as np
from tqdm import tqdm

from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.certification.derandomized_smoothing.derandomized import DeRandomizedSmoothingMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    import torchvision
    from timm.models.vision_transformer import VisionTransformer
    from art.estimators.certification.derandomized_smoothing.vision_transformers.pytorch import PyTorchVisionTransformer
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchDeRandomizedSmoothing(DeRandomizedSmoothingMixin, PyTorchClassifier):
    """
    Interface class for the two De-randomized smoothing approaches supported by ART for pytorch.

    If a regular pytorch neural network is fed in then (De)Randomized Smoothing as introduced in Levine et al. (2020)
    is used.

    Otherwise, if a timm vision transfomer is fed in then Certified Patch Robustness via Smoothed Vision Transformers
    as introduced in Salman et al. (2021) is used.
    """

    def __init__(
        self,
        model: Union[str, "VisionTransformer", "torch.nn.Module"],
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        ablation_size: int,
        algorithm: str = "salman2021",
        ablation_type: str = "column",
        replace_last_layer: Optional[bool] = None,
        drop_tokens: bool = True,
        load_pretrained: bool = True,
        optimizer: Union[type, "torch.optim.Optimizer", None] = None,
        optimizer_params: Optional[dict] = None,
        channels_first: bool = True,
        threshold: Optional[float] = None,
        logits: Optional[bool] = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
        verbose: bool = True,
        **kwargs,
    ):
        """
        Create a smoothed classifier.

        :param model: Either a CNN or a VIT. For a ViT supply a string specifying which ViT architecture to load from
                      the ViT library, or a vision transformer already created with the
                      Pytorch Image Models (timm) library. To run Levine et al. (2020) provide a regular pytorch model.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
                     categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param ablation_size: The size of the data portion to retain after ablation.
        :param algorithm: Either 'salman2021' or 'levine2020'. For salman2021 we support ViTs and CNNs. For levine2020
                          there is only CNN support.
        :param replace_last_layer: ViT Specific. If to replace the last layer of the ViT with a fresh layer
                                   matching the number of classes for the dataset to be examined.
                                   Needed if going from the pre-trained imagenet models to fine-tune
                                   on a dataset like CIFAR.
        :param drop_tokens: ViT Specific. If to drop the fully ablated tokens in the ViT
        :param load_pretrained: ViT Specific. If to load a pretrained model matching the ViT name.
                                Will only affect the ViT if a string name is passed to model rather than a ViT directly.
        :param optimizer: The optimizer used to train the classifier.
        :param ablation_type: The type of ablation to perform. Either "column", "row", or "block"
        :param threshold: Specific to Levine et al. The minimum threshold to count a prediction.
        :param logits: Specific to Levine et al. If the model returns logits or normalized probabilities
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

        import torch

        if not channels_first:
            raise ValueError("Channels must be set to first")
        logger.info("Running algorithm: %s", algorithm)

        # Default value for output shape
        output_shape = input_shape
        self.mode = None
        if importlib.util.find_spec("timm") is not None and algorithm == "salman2021":
            from timm.models.vision_transformer import VisionTransformer

            if isinstance(model, (VisionTransformer, str)):
                import timm
                from art.estimators.certification.derandomized_smoothing.vision_transformers.pytorch import (
                    PyTorchVisionTransformer,
                )

                if replace_last_layer is None:
                    raise ValueError("If using ViTs please specify if the last layer should be replaced")

                # temporarily assign the original method to tmp_func
                tmp_func = timm.models.vision_transformer._create_vision_transformer

                # overrride with ART's ViT creation function
                timm.models.vision_transformer._create_vision_transformer = self.create_vision_transformer
                if isinstance(model, str):
                    model = timm.create_model(
                        model, pretrained=load_pretrained, drop_tokens=drop_tokens, device_type=device_type
                    )
                    if replace_last_layer:
                        model.head = torch.nn.Linear(model.head.in_features, nb_classes)
                    if isinstance(optimizer, type):
                        if optimizer_params is not None:
                            optimizer = optimizer(model.parameters(), **optimizer_params)
                        else:
                            raise ValueError("If providing an optimiser please also supply its parameters")

                elif isinstance(model, VisionTransformer):
                    pretrained_cfg = model.pretrained_cfg
                    supplied_state_dict = model.state_dict()
                    supported_models = self.get_models()
                    if pretrained_cfg["architecture"] not in supported_models:
                        raise ValueError(
                            "Architecture not supported. Use PyTorchDeRandomizedSmoothing.get_models() "
                            "to get the supported model architectures."
                        )
                    model = timm.create_model(
                        pretrained_cfg["architecture"], drop_tokens=drop_tokens, device_type=device_type
                    )
                    model.load_state_dict(supplied_state_dict)
                    if replace_last_layer:
                        model.head = torch.nn.Linear(model.head.in_features, nb_classes)

                    if optimizer is not None:
                        if not isinstance(optimizer, torch.optim.Optimizer):
                            raise ValueError("Optimizer error: must be a torch.optim.Optimizer instance")

                        converted_optimizer: Union[torch.optim.Adam, torch.optim.SGD]
                        opt_state_dict = optimizer.state_dict()
                        if isinstance(optimizer, torch.optim.Adam):
                            logging.info("Converting Adam Optimiser")
                            converted_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                        elif isinstance(optimizer, torch.optim.SGD):
                            logging.info("Converting SGD Optimiser")
                            converted_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
                        else:
                            raise ValueError("Optimiser not supported for conversion")
                        converted_optimizer.load_state_dict(opt_state_dict)

                self.to_reshape = False
                if not isinstance(model, PyTorchVisionTransformer):
                    raise ValueError("Vision transformer is not of PyTorchViT. Error occurred in PyTorchViT creation.")

                if model.default_cfg["input_size"][0] != input_shape[0]:
                    raise ValueError(
                        f'ViT requires {model.default_cfg["input_size"][0]} channel input,'
                        f" but {input_shape[0]} channels were provided."
                    )

                if model.default_cfg["input_size"] != input_shape:
                    if verbose:
                        logger.warning(
                            " ViT expects input shape of: (%i, %i, %i) but (%i, %i, %i) specified as the input shape."
                            " The input will be rescaled to (%i, %i, %i)",
                            *model.default_cfg["input_size"],
                            *input_shape,
                            *model.default_cfg["input_size"],
                        )

                    self.to_reshape = True
                output_shape = model.default_cfg["input_size"]

                # set the method back to avoid unexpected side effects later on should timm need to be reused.
                timm.models.vision_transformer._create_vision_transformer = tmp_func
                self.mode = "ViT"
            else:
                if isinstance(model, torch.nn.Module):
                    self.mode = "CNN"
                    output_shape = input_shape
                    self.to_reshape = False

        elif algorithm == "levine2020":
            if ablation_type is None or threshold is None or logits is None:
                raise ValueError(
                    "If using CNN please specify if the model returns logits, "
                    " the prediction threshold, and ablation type"
                )
            self.mode = "CNN"
            # input channels are internally doubled.
            input_shape = (input_shape[0] * 2, input_shape[1], input_shape[2])
            output_shape = input_shape
            self.to_reshape = False

        if optimizer is None or isinstance(optimizer, torch.optim.Optimizer):
            super().__init__(
                model=model,
                loss=loss,
                input_shape=input_shape,
                nb_classes=nb_classes,
                optimizer=optimizer,
                channels_first=channels_first,
                clip_values=clip_values,
                preprocessing_defences=preprocessing_defences,
                postprocessing_defences=postprocessing_defences,
                preprocessing=preprocessing,
                device_type=device_type,
            )
        else:
            raise ValueError("Error occurred in optimizer creation")

        self.threshold = threshold
        self.logits = logits
        self.ablation_size = (ablation_size,)
        self.algorithm = algorithm
        self.ablation_type = ablation_type
        if verbose:
            logger.info(self.model)

        from art.estimators.certification.derandomized_smoothing.ablators.pytorch import (
            ColumnAblatorPyTorch,
            BlockAblatorPyTorch,
        )

        if TYPE_CHECKING:
            self.ablator: Union[ColumnAblatorPyTorch, BlockAblatorPyTorch]

        if self.mode is None:
            raise ValueError("Model type not recognized.")

        if ablation_type in {"column", "row"}:
            self.ablator = ColumnAblatorPyTorch(
                ablation_size=ablation_size,
                channels_first=True,
                ablation_mode=ablation_type,
                to_reshape=self.to_reshape,
                original_shape=input_shape,
                output_shape=output_shape,
                device_type=device_type,
                algorithm=algorithm,
                mode=self.mode,
            )
        elif ablation_type == "block":
            self.ablator = BlockAblatorPyTorch(
                ablation_size=ablation_size,
                channels_first=True,
                to_reshape=self.to_reshape,
                original_shape=input_shape,
                output_shape=output_shape,
                device_type=device_type,
                algorithm=algorithm,
                mode=self.mode,
            )
        else:
            raise ValueError(f"ablation_type of {ablation_type} not recognized. Must be either column, row, or block")

    @classmethod
    def get_models(cls, generate_from_null: bool = False) -> List[str]:
        """
        Return the supported model names to the user.

        :param generate_from_null: If to re-check the creation of all the ViTs in timm from scratch.
        :return: A list of compatible models
        """
        import timm
        import torch

        supported_models = [
            "vit_base_patch8_224",
            "vit_base_patch16_18x2_224",
            "vit_base_patch16_224",
            "vit_base_patch16_224_miil",
            "vit_base_patch16_384",
            "vit_base_patch16_clip_224",
            "vit_base_patch16_clip_384",
            "vit_base_patch16_gap_224",
            "vit_base_patch16_plus_240",
            "vit_base_patch16_rpn_224",
            "vit_base_patch16_xp_224",
            "vit_base_patch32_224",
            "vit_base_patch32_384",
            "vit_base_patch32_clip_224",
            "vit_base_patch32_clip_384",
            "vit_base_patch32_clip_448",
            "vit_base_patch32_plus_256",
            "vit_giant_patch14_224",
            "vit_giant_patch14_clip_224",
            "vit_gigantic_patch14_224",
            "vit_gigantic_patch14_clip_224",
            "vit_huge_patch14_224",
            "vit_huge_patch14_clip_224",
            "vit_huge_patch14_clip_336",
            "vit_huge_patch14_xp_224",
            "vit_large_patch14_224",
            "vit_large_patch14_clip_224",
            "vit_large_patch14_clip_336",
            "vit_large_patch14_xp_224",
            "vit_large_patch16_224",
            "vit_large_patch16_384",
            "vit_large_patch32_224",
            "vit_large_patch32_384",
            "vit_medium_patch16_gap_240",
            "vit_medium_patch16_gap_256",
            "vit_medium_patch16_gap_384",
            "vit_small_patch16_18x2_224",
            "vit_small_patch16_36x1_224",
            "vit_small_patch16_224",
            "vit_small_patch16_384",
            "vit_small_patch32_224",
            "vit_small_patch32_384",
            "vit_tiny_patch16_224",
            "vit_tiny_patch16_384",
        ]

        if not generate_from_null:
            return supported_models

        supported = []
        unsupported = []

        models = timm.list_models("vit_*")
        pbar = tqdm(models)

        # store in case not re-assigned in the model creation due to unsuccessful creation
        tmp_func = timm.models.vision_transformer._create_vision_transformer  # pylint: disable=W0212

        for model in pbar:
            pbar.set_description(f"Testing {model} creation")
            try:
                _ = cls(
                    model=model,
                    loss=torch.nn.CrossEntropyLoss(),
                    optimizer=torch.optim.SGD,
                    optimizer_params={"lr": 0.01},
                    input_shape=(3, 32, 32),
                    nb_classes=10,
                    ablation_size=4,
                    load_pretrained=False,
                    replace_last_layer=True,
                    verbose=False,
                )
                supported.append(model)
            except (TypeError, AttributeError):
                unsupported.append(model)
                timm.models.vision_transformer._create_vision_transformer = tmp_func  # pylint: disable=W0212

        if supported != supported_models:
            logger.warning(
                "Difference between the generated and fixed model list. Although not necessarily "
                "an error, this may point to the timm library being updated."
            )

        return supported

    @staticmethod
    def create_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> "PyTorchVisionTransformer":
        """
        Creates a vision transformer using PyTorchViT which controls the forward pass of the model

        :param variant: The name of the vision transformer to load
        :param pretrained: If to load pre-trained weights
        :return: A ViT with the required methods needed for ART
        """

        from timm.models._builder import build_model_with_cfg
        from timm.models.vision_transformer import checkpoint_filter_fn
        from art.estimators.certification.derandomized_smoothing.vision_transformers.pytorch import (
            PyTorchVisionTransformer,
        )

        return build_model_with_cfg(
            PyTorchVisionTransformer,
            variant,
            pretrained,
            pretrained_filter_fn=checkpoint_filter_fn,
            **kwargs,
        )

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        training_mode: bool = True,
        drop_last: bool = False,
        scheduler: Optional[Any] = None,
        verbose: bool = False,
        update_batchnorm: bool = True,
        batchnorm_update_epochs: int = 1,
        transform: Optional["torchvision.transforms.transforms.Compose"] = None,
        **kwargs,
    ) -> None:
        """
        Fit the classifier on the training set `(x, y)`.

        :param x: Training data.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or index labels of
                  shape (nb_samples,).
        :param batch_size: Size of batches.
        :param nb_epochs: Number of epochs to use for training.
        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.
        :param drop_last: Set to ``True`` to drop the last incomplete batch, if the dataset size is not divisible by
                          the batch size. If ``False`` and the size of dataset is not divisible by the batch size, then
                          the last batch will be smaller. (default: ``False``)
        :param scheduler: Learning rate scheduler to run at the start of every epoch.
        :param verbose: Display training progress bar.
        :param update_batchnorm: ViT specific argument.
                                 If to run the training data through the model to update any batch norm statistics prior
                                 to training. Useful on small datasets when using pre-trained ViTs.
        :param batchnorm_update_epochs: ViT specific argument. How many times to forward pass over the training data
                                        to pre-adjust the batchnorm statistics.
        :param transform: ViT specific argument. Torchvision compose of relevant augmentation transformations to apply.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch

        # Set model mode
        self._model.train(mode=training_mode)

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        if update_batchnorm and self.mode == "ViT":  # VIT specific
            self.update_batchnorm(x_preprocessed, batch_size, nb_epochs=batchnorm_update_epochs)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        num_batch = len(x_preprocessed) / float(batch_size)
        num_batch = int(np.floor(num_batch)) if drop_last else int(np.ceil(num_batch))
        ind = np.arange(len(x_preprocessed)).tolist()

        # Start training
        for _ in tqdm(range(nb_epochs)):
            # Shuffle the examples
            random.shuffle(ind)

            epoch_acc = []
            epoch_loss = []
            epoch_batch_sizes = []

            pbar = tqdm(range(num_batch), disable=not verbose)

            # Train for one epoch
            for m in pbar:
                i_batch = self.ablator.forward(np.copy(x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]))

                if transform is not None and self.mode == "ViT":  # VIT specific
                    i_batch = transform(i_batch)

                if isinstance(i_batch, np.ndarray):
                    i_batch = torch.from_numpy(i_batch).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size : (m + 1) * batch_size]]).to(self._device)

                # Zero the parameter gradients
                self._optimizer.zero_grad()

                # Perform prediction
                try:
                    model_outputs = self.model(i_batch)
                except ValueError as err:
                    if "Expected more than 1 value per channel when training" in str(err):
                        logger.exception(
                            "Try dropping the last incomplete batch by setting drop_last=True in "
                            "method PyTorchClassifier.fit."
                        )
                    raise err

                loss = self.loss(model_outputs, o_batch)
                acc = self.get_accuracy(preds=model_outputs, labels=o_batch)

                # Do training
                if self._use_amp:  # pragma: no cover
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss.backward()

                self.optimizer.step()

                epoch_acc.append(acc)
                epoch_loss.append(loss.cpu().detach().numpy())
                epoch_batch_sizes.append(len(i_batch))

                if verbose:
                    pbar.set_description(
                        f"Loss {np.average(epoch_loss, weights=epoch_batch_sizes):.3f} "
                        f"Acc {np.average(epoch_acc, weights=epoch_batch_sizes):.3f} "
                    )

            if scheduler is not None:
                scheduler.step()

    @staticmethod
    def get_accuracy(preds: Union[np.ndarray, "torch.Tensor"], labels: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Helper function to get the accuracy during training.

        :param preds: model predictions.
        :param labels: ground truth labels (not one hot).
        :return: prediction accuracy.
        """
        if not isinstance(preds, np.ndarray):
            preds = preds.detach().cpu().numpy()

        if not isinstance(labels, np.ndarray):
            labels = labels.detach().cpu().numpy()

        return np.sum(np.argmax(preds, axis=1) == labels) / len(labels)

    def update_batchnorm(self, x: np.ndarray, batch_size: int, nb_epochs: int = 1) -> None:
        """
        Method to update the batchnorm of a neural network on small datasets when it was pre-trained

        :param x: Training data.
        :param batch_size: Size of batches.
        :param nb_epochs: How many times to forward pass over the input data
        """
        import torch

        if self.mode != "ViT":
            raise ValueError("Accessing a ViT specific functionality while running in CNN mode")

        self.model.train()

        ind = np.arange(len(x))
        num_batch = int(len(x) / float(batch_size))

        with torch.no_grad():
            for _ in tqdm(range(nb_epochs)):
                for m in tqdm(range(num_batch)):
                    i_batch = self.ablator.forward(
                        np.copy(x[ind[m * batch_size : (m + 1) * batch_size]]), column_pos=random.randint(0, x.shape[3])
                    )
                    _ = self.model(i_batch)

    def eval_and_certify(
        self,
        x: np.ndarray,
        y: np.ndarray,
        size_to_certify: int,
        batch_size: int = 128,
        verbose: bool = True,
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Evaluates the ViT's normal and certified performance over the supplied data.

        :param x: Evaluation data.
        :param y: Evaluation labels.
        :param size_to_certify: The size of the patch to certify against.
                                If not provided will default to the ablation size.
        :param batch_size: batch size when evaluating.
        :param verbose: If to display the progress bar
        :return: The accuracy and certified accuracy over the dataset
        """
        import torch

        self.model.eval()
        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        num_batch = int(np.ceil(len(x_preprocessed) / float(batch_size)))
        pbar = tqdm(range(num_batch), disable=not verbose)
        accuracy = torch.tensor(0.0).to(self._device)
        cert_sum = torch.tensor(0.0).to(self._device)
        n_samples = 0

        with torch.no_grad():
            for m in pbar:
                if m == (num_batch - 1):
                    i_batch = np.copy(x_preprocessed[m * batch_size :])
                    o_batch = y_preprocessed[m * batch_size :]
                else:
                    i_batch = np.copy(x_preprocessed[m * batch_size : (m + 1) * batch_size])
                    o_batch = y_preprocessed[m * batch_size : (m + 1) * batch_size]

                pred_counts = np.zeros((len(i_batch), self.nb_classes))
                if self.ablation_type in {"column", "row"}:
                    for pos in range(i_batch.shape[-1]):
                        ablated_batch = self.ablator.forward(i_batch, column_pos=pos)
                        # Perform prediction
                        model_outputs = self.model(ablated_batch)

                        if self.algorithm == "salman2021":
                            pred_counts[np.arange(0, len(i_batch)), model_outputs.argmax(dim=-1).cpu()] += 1
                        else:
                            if self.logits:
                                model_outputs = torch.nn.functional.softmax(model_outputs, dim=1)
                            model_outputs = model_outputs >= self.threshold
                            pred_counts += model_outputs.cpu().numpy()

                else:
                    for column_pos in range(i_batch.shape[-1]):
                        for row_pos in range(i_batch.shape[-2]):
                            ablated_batch = self.ablator.forward(i_batch, column_pos=column_pos, row_pos=row_pos)
                            model_outputs = self.model(ablated_batch)

                            if self.algorithm == "salman2021":
                                pred_counts[np.arange(0, len(i_batch)), model_outputs.argmax(dim=-1).cpu()] += 1
                            else:
                                if self.logits:
                                    model_outputs = torch.nn.functional.softmax(model_outputs, dim=1)
                                model_outputs = model_outputs >= self.threshold
                                pred_counts += model_outputs.cpu().numpy()

                _, cert_and_correct, top_predicted_class = self.ablator.certify(
                    pred_counts, size_to_certify=size_to_certify, label=o_batch
                )
                cert_sum += torch.sum(cert_and_correct)
                o_batch = torch.from_numpy(o_batch).to(self.device)
                accuracy += torch.sum(top_predicted_class == o_batch)
                n_samples += len(cert_and_correct)

                pbar.set_description(f"Normal Acc {accuracy / n_samples:.3f} " f"Cert Acc {cert_sum / n_samples:.3f}")

        return (accuracy / n_samples), (cert_sum / n_samples)

    def _predict_classifier(
        self, x: Union[np.ndarray, "torch.Tensor"], batch_size: int, training_mode: bool, **kwargs
    ) -> np.ndarray:
        import torch

        if isinstance(x, torch.Tensor):
            x_numpy = x.cpu().numpy()

        outputs = PyTorchClassifier.predict(
            self, x=x_numpy, batch_size=batch_size, training_mode=training_mode, **kwargs
        )

        if self.algorithm == "levine2020":
            if not self.logits:
                return np.asarray((outputs >= self.threshold))
            return np.asarray(
                (torch.nn.functional.softmax(torch.from_numpy(outputs), dim=1) >= self.threshold).type(torch.int)
            )
        return outputs

    def predict(self, x: np.ndarray, batch_size: int = 128, training_mode: bool = False, **kwargs) -> np.ndarray:
        """
        Performs cumulative predictions over every ablation location

        :param x: Unablated image
        :param batch_size: the batch size for the prediction
        :param training_mode: if to run the classifier in training mode
        :return: cumulative predictions after sweeping over all the ablation configurations.
        """
        if self._channels_first:
            columns_in_data = x.shape[-1]
            rows_in_data = x.shape[-2]
        else:
            columns_in_data = x.shape[-2]
            rows_in_data = x.shape[-3]

        if self.ablation_type in {"column", "row"}:
            if self.ablation_type == "column":
                ablate_over_range = columns_in_data
            else:
                # image will be transposed, so loop over the number of rows
                ablate_over_range = rows_in_data

            for ablation_start in range(ablate_over_range):
                ablated_x = self.ablator.forward(np.copy(x), column_pos=ablation_start)
                if ablation_start == 0:
                    preds = self._predict_classifier(
                        ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                    )
                else:
                    preds += self._predict_classifier(
                        ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                    )
        elif self.ablation_type == "block":
            for xcorner in range(rows_in_data):
                for ycorner in range(columns_in_data):
                    ablated_x = self.ablator.forward(np.copy(x), row_pos=xcorner, column_pos=ycorner)
                    if ycorner == 0 and xcorner == 0:
                        preds = self._predict_classifier(
                            ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                        )
                    else:
                        preds += self._predict_classifier(
                            ablated_x, batch_size=batch_size, training_mode=training_mode, **kwargs
                        )

        return preds
