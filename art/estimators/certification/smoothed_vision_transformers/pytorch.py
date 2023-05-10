from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
from typing import List, Optional, Tuple, Union, Any, TYPE_CHECKING
import random
import torch
import copy

from timm.models.vision_transformer import VisionTransformer

import numpy as np
from tqdm import tqdm

from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.certification.smoothed_vision_transformers.smooth_vit import ColumnAblator
from art.utils import check_and_transform_label_format

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatchEmbed(torch.nn.Module):
    """ Image to Patch Embedding

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
    SOFTWARE.

    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def create(self, patch_size=None, embed_dim=None, **kwargs):

        if patch_size is not None:
            self.patch_size = patch_size
        if embed_dim is not None:
            self.embed_dim = embed_dim

        self.proj = torch.nn.Conv2d(in_channels=1,
                                    out_channels=self.embed_dim,
                                    kernel_size=self.patch_size,
                                    stride=self.patch_size,
                                    bias=False)

        w_shape = self.proj.weight.shape
        self.proj.weight = torch.nn.Parameter(torch.ones(w_shape).to(device))

    def forward(self, x):
        with torch.no_grad():
            x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ArtViT(VisionTransformer):

    # Make as a class attribute to avoid being included in the
    # state dictionaries of the ViT Model.
    ablation_mask_embedder = PatchEmbed(in_chans=1)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ablation_mask_embedder.create(**kwargs)

    @staticmethod
    def drop_tokens(x, indexes):
        """
        Drops the tokens which correspond to fully masked inputs
        :param x: Input data in .... format
        :param indexes: positions to be ablated
        return
        """
        x_no_cl, cls_token = x[:, 1:], x[:, 0:1]
        shape = x_no_cl.shape

        # reshape to temporarily remove batch
        x_no_cl = torch.reshape(x_no_cl, shape=(-1, shape[-1]))
        indexes = torch.reshape(indexes, shape=(-1,))
        indexes = (indexes == True).nonzero(as_tuple=True)[0]

        x_no_cl = torch.index_select(x_no_cl, dim=0, index=indexes)
        x_no_cl = torch.reshape(x_no_cl, shape=(shape[0], -1, shape[-1]))
        return torch.cat((cls_token, x_no_cl), dim=1)

    def forward_features(self, x):
        """
        The forward pass of the ViT.
        #TODO! check for 1 channel inputs!

        :param x: Input data.

        """
        drop_tokens = True

        if x.shape[1] == 4:
            x, ablation_mask = x[:, :3], x[:, 3:4]

        x = self.patch_embed(x)
        x = self._pos_embed(x)

        if drop_tokens:
            ones = self.ablation_mask_embedder(ablation_mask)
            to_drop = torch.sum(ones, dim=2)
            indexes = torch.gt(torch.where(to_drop > 1, 1, 0), 0)

            check_i = indexes[0]
            check_val = to_drop[0]
            for i, s in zip(indexes, to_drop):
                if not torch.equal(check_i, i):
                    for ci, ei, val, cval in zip(check_i, i, s, check_val):
                        print(f'{ci} with {cval} vs {ei} with {val}')
                    sys.exit()

            x = self.drop_tokens(x, indexes)

        x = self.blocks(x)
        return self.norm(x)


class PyTorchSmoothedViT(PyTorchClassifier):
    def __init__(
        self,
        model: ["VisionTransformer", str],
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        ablation_type: str,
        ablation_size: int,
        threshold: float,
        optimizer: Optional["torch.optim.Optimizer"] = None,  # type: ignore
        optimizer_params: Optional[dict] = None,
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
        load_pretrained: bool = True,
    ):
        """
        Create a smoothed ViT classifier.

        :param model: Either a string specifying which ViT architecture to load, or a vision transformer already
                      created with the Pytorch Image Models (timm) library.
        :param loss: The loss function for which to compute gradients for training. The target label must be raw
               categorical, i.e. not converted to one-hot encoding.
        :param input_shape: The shape of one input instance.
        :param nb_classes: The number of classes of the model.
        :param ablation_type: The type of ablation to perform, must be either "column" or "block"
        :param ablation_size: The size of the data portion to retain after ablation. Will be a column of size N for
                              "column" ablation type or a NxN square for ablation of type "block"
        :param threshold: The minimum threshold to count a prediction.
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

        timm.models.vision_transformer._create_vision_transformer = self.art_create_vision_transformer

        if type(model) is str:
            model = timm.create_model(model, pretrained=load_pretrained)
            model.head = torch.nn.Linear(model.head.in_features, nb_classes)
            # TODO: enable users to pass in opt hyperparameters
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
            optimizer = optimizer(model.parameters(), **optimizer_params)

        else:
            pretrained_cfg = model.pretrained_cfg
            supplied_state_dict = model.state_dict()
            model = timm.create_model(pretrained_cfg['vit_small_patch16_224'], pretrained=load_pretrained)
            model.load_state_dict(torch.load(supplied_state_dict))
            model.head = torch.nn.Linear(model.head.in_features, nb_classes)

            if optimizer is not None:
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

        self.ablation_type = ablation_type
        self.ablation_size = ablation_size,
        self.threshold = threshold

        print(self.model)
        self.ablator = ColumnAblator(ablation_size=ablation_size,
                                     channels_first=True,
                                     row_ablation_mode=False)

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
            ArtViT, variant, pretrained,
            pretrained_filter_fn=checkpoint_filter_fn,
            **kwargs,
        )

    def update_batchnorm(self, x: np.ndarray, batch_size: int) -> None:
        """
        Method to update the batchnorm of a ViT on small datasets
        :param x:
        :param batch_size: Size of batches.
        """
        import random
        import time

        self.model.train()

        ind = np.arange(len(x))
        num_batch = int(len(x) / float(batch_size))

        print('updating batchnorm')
        s = time.time()
        with torch.no_grad():
            for _ in tqdm(range(1)):
                for m in tqdm(range(num_batch)):
                    i_batch = torch.from_numpy(np.copy(x[ind[m * batch_size: (m + 1) * batch_size]])).to(device)
                    i_batch = self.ablator.forward(i_batch, column_pos=random.randint(0, x.shape[3]))
                    _ = self.model(i_batch)
        print('total time taken is ', time.time() - s)

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
        :param update_batchnorm: ...
        :param verbose: ...
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch

        # Set model mode
        self._model.train(mode=training_mode)

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided.")

        import torchvision.transforms as transforms

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip()
            ]
        )

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        if update_batchnorm:
            self.update_batchnorm(x_preprocessed, batch_size)

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

            pbar = tqdm(range(num_batch), disable=not verbose)

            # Train for one epoch
            for m in pbar:
                i_batch = torch.from_numpy(np.copy(x_preprocessed[ind[m * batch_size: (m + 1) * batch_size]])).to(self._device)
                i_batch = transform(i_batch)
                i_batch = self.ablator.forward(i_batch, column_pos=random.randint(0, x.shape[3]))

                o_batch = torch.from_numpy(y_preprocessed[ind[m * batch_size: (m + 1) * batch_size]]).to(
                    self._device)

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
                # Form the loss function
                # print('the model outputs are ', model_outputs.shape)
                loss = self.loss(model_outputs, o_batch)
                acc = self.get_accuracy(preds=model_outputs, labels=o_batch)
                epoch_acc.append(acc)
                epoch_loss.append(loss)

                # Do training
                if self._use_amp:  # pragma: no cover
                    from apex import amp  # pylint: disable=E0611

                    with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss.backward()

                self.optimizer.step()

                if verbose:
                    pbar.set_description(
                        f"Loss {torch.mean(torch.stack(epoch_loss)):.2f}"
                        f" Acc {np.mean(epoch_acc):.2f}"
                    )

            if scheduler is not None:
                scheduler.step()

    def eval_and_certify(self, x: np.ndarray, y: np.ndarray, batch_size: int = 128):
        """
        Evaluates the ViT's normal and certified performance over the supplied data
        :param x: Evaluation data
        :param y: Evaluation labels
        :param batch_size: batch size when evaluating
        """

        self.model.eval()
        drop_last = True
        verbose = True
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
        pbar = tqdm(range(num_batch), disable=not verbose)
        accuracy = []
        cert_acc = []
        with torch.no_grad():
            for m in pbar:
                i_batch = torch.from_numpy(np.copy(x_preprocessed[m * batch_size: (m + 1) * batch_size])).to(self._device)
                o_batch = torch.from_numpy(y_preprocessed[m * batch_size: (m + 1) * batch_size]).to(self._device)
                predictions = []
                pred_counts = torch.zeros((batch_size, self.nb_classes)).to(self._device)
                for pos in range(i_batch.shape[-1]):

                    ablated_batch = self.ablator.forward(i_batch, column_pos=pos)

                    # Perform prediction
                    model_outputs = self.model(ablated_batch)
                    pred_counts[np.arange(0, batch_size), model_outputs.argmax(dim=-1)] += 1
                    predictions.append(model_outputs)

                cert, cert_and_correct, top_predicted_class = self.ablator.certify(pred_counts, size_to_certify=4, label=o_batch)
                cert_acc.append(torch.sum(cert_and_correct) / batch_size)
                acc = torch.sum(top_predicted_class == o_batch) / batch_size
                accuracy.append(acc)

                print('Normal Acc: ', torch.mean(torch.stack(accuracy)))
                print('Cert Normal Acc: ', torch.mean(torch.stack(cert_acc)))

    @staticmethod
    def get_accuracy(preds: Union[np.ndarray, "torch.Tensor"], labels: Union[np.ndarray, "torch.Tensor"]) -> np.ndarray:
        """
        Helper function to print out the accuracy during training

        :param preds: (concrete) model predictions
        :param labels: ground truth labels (not one hot)
        :return: prediction accuracy
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()

        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()

        return np.sum(np.argmax(preds, axis=1) == labels) / len(labels)