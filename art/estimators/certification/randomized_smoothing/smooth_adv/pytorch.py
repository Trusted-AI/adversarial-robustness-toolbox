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
This module implements SmoothAdv applied to classifier predictions.

| Paper link: https://arxiv.org/abs/1906.04584
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

from tqdm.auto import trange
import numpy as np

from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.estimators.certification.randomized_smoothing.pytorch import PyTorchRandomizedSmoothing
from art.estimators.classification.pytorch import PyTorchClassifier
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchSmoothAdv(PyTorchRandomizedSmoothing):
    """
    Implementation of SmoothAdv training, as introduced in Salman et al. (2019)

    | Paper link: https://arxiv.org/abs/1906.04584
    """

    estimator_params = PyTorchRandomizedSmoothing.estimator_params + [
        "epsilon",
        "num_noise_vec",
        "num_steps",
        "warmup",
    ]

    def __init__(
        self,
        model: "torch.nn.Module",
        loss: "torch.nn.modules.loss._Loss",
        input_shape: Tuple[int, ...],
        nb_classes: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,
        channels_first: bool = True,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = (0.0, 1.0),
        device_type: str = "gpu",
        sample_size: int = 32,
        scale: float = 0.1,
        alpha: float = 0.001,
        epsilon: float = 1.0,
        num_noise_vec: int = 1,
        num_steps: int = 10,
        warmup: int = 1,
    ) -> None:
        """
        Create a SmoothAdv classifier.

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
        :param sample_size: Number of samples for smoothing.
        :param scale: Standard deviation of Gaussian noise added.
        :param alpha: The failure probability of smoothing.
        :param epsilon: The maximum perturbation that can be induced.
        :param num_noise_vec: The number of noise vectors.
        :param num_steps: The number of attack updates.
        :param warmup: The warm-up strategy that is gradually increased up to the original value.
        """
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
            sample_size=sample_size,
            scale=scale,
            alpha=alpha,
        )
        self.epsilon = epsilon
        self.num_noise_vec = num_noise_vec
        self.num_steps = num_steps
        self.warmup = warmup

        classifier = PyTorchClassifier(
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
        self.attack = ProjectedGradientDescent(classifier, eps=self.epsilon, max_iter=1, verbose=False)

    def fit(  # pylint: disable=W0221
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 128,
        nb_epochs: int = 10,
        training_mode: bool = True,
        drop_last: bool = False,
        scheduler: Optional["torch.optim.lr_scheduler._LRScheduler"] = None,
        verbose: bool = False,
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
        :param verbose: Display the training progress bar.
        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch
               and providing it takes no effect.
        """
        import torch
        from torch.utils.data import TensorDataset, DataLoader

        # Set model mode
        self._model.train(mode=training_mode)

        if self._optimizer is None:  # pragma: no cover
            raise ValueError("An optimizer is needed to train the model, but none for provided")

        y = check_and_transform_label_format(y, nb_classes=self.nb_classes)

        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=True)

        # Check label shape
        y_preprocessed = self.reduce_labels(y_preprocessed)

        # Create dataloader
        x_tensor = torch.from_numpy(x_preprocessed)
        y_tensor = torch.from_numpy(y_preprocessed)
        dataset = TensorDataset(x_tensor, y_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last)

        # Start training
        for epoch in trange(nb_epochs, disable=not verbose):
            self.attack.norm = min(self.epsilon, (epoch + 1) * self.epsilon / self.warmup)

            for x_batch, y_batch in dataloader:
                mini_batch_size = len(x_batch) // self.num_noise_vec

                for mini_batch in range(self.num_noise_vec):
                    # Create mini batch
                    inputs = x_batch[mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size]
                    labels = y_batch[mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size]

                    # Move inputs to GPU
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    noise = torch.randn_like(inputs) * self.scale

                    # Attack and find adversarial examples
                    self.model.eval()
                    original_inputs = inputs.cpu().detach().numpy()
                    noise_for_attack = noise.cpu().detach().numpy()
                    perturbation_delta = np.zeros_like(original_inputs)
                    for _ in range(self.num_steps):
                        perturbed_inputs = original_inputs + perturbation_delta
                        adv_ex = self.attack.generate(perturbed_inputs + noise_for_attack)
                        perturbation_delta = adv_ex - perturbed_inputs - noise_for_attack

                    # Update perturbed inputs after last iteration
                    perturbed_inputs = original_inputs + perturbation_delta
                    self.model.train()
                    noisy_inputs = torch.from_numpy(perturbed_inputs).to(self.device) + noise

                    targets = labels.unsqueeze(1).repeat(1, self.num_noise_vec).reshape(-1, 1).squeeze()
                    outputs = self.model(noisy_inputs)
                    loss = self.loss(outputs, targets)

                    # Compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            if scheduler is not None:
                scheduler.step()
