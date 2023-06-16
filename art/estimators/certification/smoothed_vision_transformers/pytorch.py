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
This module implements Certified Patch Robustness via Smoothed Vision Transformers

| Paper link Accepted version:
    https://openaccess.thecvf.com/content/CVPR2022/papers/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.pdf

| Paper link Arxiv version (more detail): https://arxiv.org/pdf/2110.07719.pdf
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
import random

import numpy as np
from timm.models.vision_transformer import VisionTransformer
import torch
from tqdm import tqdm

from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.certification.smoothed_vision_transformers.smooth_vit import ColumnAblator
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    import torchvision
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatchEmbed(torch.nn.Module):
    """
    Image to Patch Embedding

    Class adapted from the implementation in https://github.com/MadryLab/smoothed-vit

    Original License:

    MIT License

    Copyright (c) 2021 Madry Lab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 1, embed_dim: int = 768):
        """
        Specifies the configuration for the convolutional layer.

        :param patch_size: The patch size used by the ViT.
        :param in_channels: Number of input channels.
        :param embed_dim: The embedding dimension used by the ViT.
        """
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj: Optional[torch.nn.Conv2d] = None

    def create(self, patch_size=None, embed_dim=None, device="cpu", **kwargs) -> None:  # pylint: disable=W0613
        """
        Creates a convolution that mimics the embedding layer to be used for the ablation mask to
        track where the image was ablated.

        :param patch_size: The patch size used by the ViT
        :param embed_dim: The embedding dimension used by the ViT
        :param device: Which device to set the emdedding layer to.
        :param kwargs: Handles the remaining kwargs from the ViT configuration.
        """

        if patch_size is not None:
            self.patch_size = patch_size
        if embed_dim is not None:
            self.embed_dim = embed_dim

        self.proj = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        w_shape = self.proj.weight.shape
        self.proj.weight = torch.nn.Parameter(torch.ones(w_shape).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the embedder. We are simply tracking the positions of the ablation mask so no gradients
        are required.

        :param x: Input data corresponding to the ablation mask
        :return: The embedded input
        """
        if self.proj is not None:
            with torch.no_grad():
                x = self.proj(x).flatten(2).transpose(1, 2)
            return x
        raise ValueError("Projection layer not yet created.")


class ArtViT(VisionTransformer):
    """
    Art class inheriting from VisionTransformer to control the forward pass of the ViT.
    """

    # Make as a class attribute to avoid being included in the
    # state dictionaries of the ViT Model.
    ablation_mask_embedder = PatchEmbed(in_channels=1)

    def __init__(self, **kwargs):
        """
        Create a ArtViT instance
        :param kwargs: keyword arguments required to create the mask embedder and the vision transformer class
        Must contain ...
        """
        self.to_drop_tokens = kwargs["drop_tokens"]

        if kwargs["device_type"] == "cpu" or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self.device = torch.device(f"cuda:{cuda_idx}")

        del kwargs["drop_tokens"]
        del kwargs["device_type"]

        super().__init__(**kwargs)
        self.ablation_mask_embedder.create(device=self.device, **kwargs)

        self.in_chans = kwargs["in_chans"]
        self.img_size = kwargs["img_size"]

    @staticmethod
    def drop_tokens(x: torch.Tensor, indexes: torch.Tensor) -> torch.Tensor:
        """
        Drops the tokens which correspond to fully masked inputs

        :param x: Input data
        :param indexes: positions to be ablated
        :return: Input with tokens dropped where the input was fully ablated.
        """
        x_no_cl, cls_token = x[:, 1:], x[:, 0:1]
        shape = x_no_cl.shape

        # reshape to temporarily remove batch
        x_no_cl = torch.reshape(x_no_cl, shape=(-1, shape[-1]))
        indexes = torch.reshape(indexes, shape=(-1,))
        indexes = indexes.nonzero(as_tuple=True)[0]
        x_no_cl = torch.index_select(x_no_cl, dim=0, index=indexes)
        x_no_cl = torch.reshape(x_no_cl, shape=(shape[0], -1, shape[-1]))
        return torch.cat((cls_token, x_no_cl), dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the ViT.

        :param x: Input data.
        :return: The input processed by the ViT backbone
        """

        ablated_input = False
        if x.shape[1] == self.in_chans + 1:
            ablated_input = True

        if ablated_input:
            x, ablation_mask = x[:, : self.in_chans], x[:, self.in_chans : self.in_chans + 1]

        x = self.patch_embed(x)
        x = self._pos_embed(x)

        if self.to_drop_tokens and ablated_input:
            ones = self.ablation_mask_embedder(ablation_mask)
            to_drop = torch.sum(ones, dim=2)
            indexes = torch.gt(torch.where(to_drop > 1, 1, 0), 0)
            x = self.drop_tokens(x, indexes)

        x = self.norm_pre(x)
        x = self.blocks(x)
        return self.norm(x)


class PyTorchSmoothedViT(PyTorchClassifier):
    """
    Implementation of Certified Patch Robustness via Smoothed Vision Transformers

    | Paper link Accepted version:
        https://openaccess.thecvf.com/content/CVPR2022/papers/Salman_Certified_Patch_Robustness_via_Smoothed_Vision_Transformers_CVPR_2022_paper.pdf

    | Paper link Arxiv version (more detail): https://arxiv.org/pdf/2110.07719.pdf
    """

    def __init__(
        self,
        model: Union[VisionTransformer, str],
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        ablation_size: int,
        replace_last_layer: bool,
        drop_tokens: bool = True,
        load_pretrained: bool = True,
        optimizer: Union[type, "torch.optim.Optimizer", None] = None,
        optimizer_params: Optional[dict] = None,
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
        verbose: bool = True,
    ):
        """
        Create a smoothed ViT classifier.

        :param model: Either a string specifying which ViT architecture to load, or a vision transformer already
                      created with the Pytorch Image Models (timm) library.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param ablation_size: The size of the data portion to retain after ablation.
        :param replace_last_layer: If to replace the last layer of the ViT with a fresh layer matching the number
                           of classes for the dataset to be examined. Needed if going from the pre-trained
                           imagenet models to fine-tune on a dataset like CIFAR.
        :param drop_tokens: If to drop the fully ablated tokens in the ViT
        :param load_pretrained: If to load a pretrained model matching the ViT name. Will only affect the ViT if a
                        string name is passed to model rather than a ViT directly.
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
        import timm

        # temporarily assign the original method to tmp_func
        tmp_func = timm.models.vision_transformer._create_vision_transformer

        # overrride with ART's ViT creation function
        timm.models.vision_transformer._create_vision_transformer = self.art_create_vision_transformer
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
                    "Architecture not supported. Use PyTorchSmoothedViT.get_models() "
                    "to get the supported model architectures."
                )
            model = timm.create_model(pretrained_cfg["architecture"], drop_tokens=drop_tokens, device_type=device_type)
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
        if not isinstance(model, ArtViT):
            raise ValueError("Vision transformer is not of ArtViT. Error occurred in ArtViT creation.")

        if model.default_cfg["input_size"][0] != input_shape[0]:
            raise ValueError(
                f'ViT requires {model.default_cfg["input_size"][0]} channel input,'
                f" but {input_shape[0]} channels were provided."
            )

        if model.default_cfg["input_size"] != input_shape:
            if verbose:
                logger.warning(
                    f"ViT expects input shape of {model.default_cfg['input_size']}, "
                    f"but {input_shape} specified as the input shape. "
                    f"The input will be rescaled to {model.default_cfg['input_size']}"
                )
            self.to_reshape = True

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

        self.ablation_size = (ablation_size,)

        if verbose:
            logger.info(self.model)

        self.ablator = ColumnAblator(
            ablation_size=ablation_size,
            channels_first=True,
            to_reshape=self.to_reshape,
            original_shape=input_shape,
            output_shape=model.default_cfg["input_size"],
            device_type=device_type,
        )

        # set the method back to avoid unexpected side effects later on should timm need to be reused.
        timm.models.vision_transformer._create_vision_transformer = tmp_func

    @classmethod
    def get_models(cls, generate_from_null: bool = False) -> List[str]:
        """
        Return the supported model names to the user.

        :param generate_from_null: If to re-check the creation of all the ViTs in timm from scratch.
        :return: A list of compatible models
        """
        import timm

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
        for model in models:
            logger.info(f"Testing {model} creation")
            try:
                _ = PyTorchSmoothedViT(
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

        if supported != supported_models:
            logger.warning(
                "Difference between the generated and fixed model list. Although not necessarily "
                "an error, this may point to the timm library being updated."
            )

        return supported

    @staticmethod
    def art_create_vision_transformer(variant: str, pretrained: bool = False, **kwargs) -> ArtViT:
        """
        Creates a vision transformer using ArtViT which controls the forward pass of the model

        :param variant: The name of the vision transformer to load
        :param pretrained: If to load pre-trained weights
        :return: A ViT with the required methods needed for ART
        """

        from timm.models._builder import build_model_with_cfg
        from timm.models.vision_transformer import checkpoint_filter_fn

        return build_model_with_cfg(
            ArtViT,
            variant,
            pretrained,
            pretrained_filter_fn=checkpoint_filter_fn,
            **kwargs,
        )

    def update_batchnorm(self, x: np.ndarray, batch_size: int, nb_epochs: int = 1) -> None:
        """
        Method to update the batchnorm of a ViT on small datasets

        :param x: Training data.
        :param batch_size: Size of batches.
        :param nb_epochs: How many times to forward pass over the input data
        """

        self.model.train()

        ind = np.arange(len(x))
        num_batch = int(len(x) / float(batch_size))

        with torch.no_grad():
            for _ in tqdm(range(nb_epochs)):
                for m in tqdm(range(num_batch)):
                    i_batch = torch.from_numpy(np.copy(x[ind[m * batch_size : (m + 1) * batch_size]])).to(self.device)
                    i_batch = self.ablator.forward(i_batch, column_pos=random.randint(0, x.shape[3]))
                    _ = self.model(i_batch)

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        training_mode: bool = True,
        drop_last: bool = False,
        scheduler: Optional[Any] = None,
        update_batchnorm: bool = True,
        batchnorm_update_epochs: int = 1,
        transform: Optional["torchvision.transforms.transforms.Compose"] = None,
        verbose: bool = True,
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
        :param update_batchnorm: if to run the training data through the model to update any batch norm statistics prior
        to training. Useful on small datasets when using pre-trained ViTs.
        :param batchnorm_update_epochs: how many times to forward pass over the training data
                                        to pre-adjust the batchnorm statistics.
        :param transform: Torchvision compose of relevant augmentation transformations to apply.
        :param verbose: if to display training progress bars
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """

        # Set model mode
        self._model.train(mode=training_mode)

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        if update_batchnorm:
            self.update_batchnorm(x_preprocessed, batch_size, nb_epochs=batchnorm_update_epochs)

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

            epoch_acc = []
            epoch_loss = []
            epoch_batch_sizes = []

            pbar = tqdm(range(num_batch), disable=not verbose)

            # Train for one epoch
            for m in pbar:
                i_batch = torch.from_numpy(np.copy(x_preprocessed[ind[m * batch_size : (m + 1) * batch_size]])).to(
                    self._device
                )
                if transform is not None:
                    i_batch = transform(i_batch)
                i_batch = self.ablator.forward(i_batch, column_pos=random.randint(0, x.shape[3]))

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
                    i_batch = torch.from_numpy(np.copy(x_preprocessed[m * batch_size :])).to(self._device)
                    o_batch = torch.from_numpy(y_preprocessed[m * batch_size :]).to(self._device)
                else:
                    i_batch = torch.from_numpy(np.copy(x_preprocessed[m * batch_size : (m + 1) * batch_size])).to(
                        self._device
                    )
                    o_batch = torch.from_numpy(y_preprocessed[m * batch_size : (m + 1) * batch_size]).to(self._device)

                predictions = []
                pred_counts = torch.zeros((len(i_batch), self.nb_classes)).to(self._device)
                for pos in range(i_batch.shape[-1]):
                    ablated_batch = self.ablator.forward(i_batch, column_pos=pos)

                    # Perform prediction
                    model_outputs = self.model(ablated_batch)
                    pred_counts[np.arange(0, len(i_batch)), model_outputs.argmax(dim=-1)] += 1
                    predictions.append(model_outputs)

                _, cert_and_correct, top_predicted_class = self.ablator.certify(
                    pred_counts, size_to_certify=size_to_certify, label=o_batch
                )
                cert_sum += torch.sum(cert_and_correct)
                accuracy += torch.sum(top_predicted_class == o_batch)
                n_samples += len(cert_and_correct)

                pbar.set_description(f"Normal Acc {accuracy / n_samples:.3f} " f"Cert Acc {cert_sum / n_samples:.3f}")

        return (accuracy / n_samples), (cert_sum / n_samples)

    @staticmethod
    def get_accuracy(preds: Union[np.ndarray, "torch.Tensor"], labels: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Helper function to get the accuracy during training.

        :param preds: model predictions.
        :param labels: ground truth labels (not one hot).
        :return: prediction accuracy.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        return np.sum(np.argmax(preds, axis=1) == labels) / len(labels)
