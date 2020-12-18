# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2020
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
This module implements the adversarial and imperceptible attack on automatic speech recognition systems of Qin et al.
(2019). It generates an adversarial audio example.

| Paper link: http://proceedings.mlr.press/v97/qin19a.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import scipy.signal as ss

from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin, NeuralNetworkMixin
from art.estimators.pytorch import PyTorchEstimator
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator
from art.utils import pad_sequence_input

if TYPE_CHECKING:
    from tensorflow.compat.v1 import Tensor
    from torch import Tensor as PTensor

    from art.utils import SPEECH_RECOGNIZER_TYPE

logger = logging.getLogger(__name__)


class ImperceptibleASR(EvasionAttack):
    """
    Implementation of the imperceptible attack against a speech recognition model.

    | Paper link: http://proceedings.mlr.press/v97/qin19a.html
    """

    attack_params = EvasionAttack.attack_params + [
        "masker",
        "eps",
        "learning_rate_1",
        "max_iter_1",
        "alpha",
        "learning_rate_2",
        "max_iter_2",
        "batch_size",
    ]

    _estimator_requirements = (NeuralNetworkMixin, LossGradientsMixin, BaseEstimator, SpeechRecognizerMixin)

    def __init__(
        self,
        estimator: "SPEECH_RECOGNIZER_TYPE",
        masker: Optional["PsychoacousticMasker"],
        eps: float = 2000.0,
        learning_rate_1: float = 100.0,
        max_iter_1: int = 1000,
        alpha: float = 0.05,
        learning_rate_2: float = 1.0,
        max_iter_2: int = 4000,
        batch_size: int = 16,
    ) -> None:
        """
        Create an instance of the :class:`.ImperceptibleASR`.

        :param estimator: A trained speech recognition estimator.
        :param masker: A Psychoacoustic masker.
        :param eps: Initial max norm bound for adversarial perturbation.
        :param learning_rate_1: Learning rate for stage 1 of attack.
        :param max_iter_1: Number of iterations for stage 1 of attack.
        :param alpha: Initial alpha value for balancing stage 2 loss.
        :param learning_rate_2: Learning rate for stage 2 of attack.
        :param max_iter_2: Number of iterations for stage 2 of attack.
        :param batch_size: Batch size.
        """

        # Super initialization
        super().__init__(estimator=estimator)
        self.masker = masker
        self.eps = eps
        self.learning_rate_1 = learning_rate_1
        self.max_iter_1 = max_iter_1
        self.alpha = alpha
        self.learning_rate_2 = learning_rate_2
        self.max_iter_2 = max_iter_2
        self._targeted = True
        self.batch_size = batch_size
        self._check_params()

        # init some aliases
        self._window_size = masker.window_size
        self._hop_size = masker.hop_size
        self._sample_rate = masker.sample_rate

        if isinstance(self.estimator, TensorFlowV2Estimator):
            import tensorflow.compat.v1 as tf1

            # set framework attribute
            self._framework = "tensorflow"

            # disable eager execution and use tensorflow.compat.v1 API, e.g. Lingvo uses TF2v1 AP
            tf1.disable_eager_execution()

            # TensorFlow placeholders
            self._delta = tf1.placeholder(tf1.float32, shape=[None, None], name="art_delta")
            self._power_spectral_density_maximum_tf = tf1.placeholder(
                tf1.float32, shape=[None, None], name="art_psd_max"
            )
            self._masking_threshold_tf = tf1.placeholder(
                tf1.float32, shape=[None, None, None], name="art_masking_threshold"
            )
            # TensorFlow loss gradient ops
            self._loss_gradient_masking_threshold_op_tf = self._loss_gradient_masking_threshold_tf(
                self._delta, self._power_spectral_density_maximum_tf, self._masking_threshold_tf
            )

        elif isinstance(self.estimator, PyTorchEstimator):
            # set framework attribute
            self._framework = "pytorch"
        else:
            # set framework attribute
            self._framework = None

    def generate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate imperceptible, adversarial examples.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different
            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: An array holding the adversarial examples.
        """
        nb_samples = x.shape[0]

        x_imperceptible = [None] * nb_samples

        nb_batches = int(np.ceil(nb_samples / float(self.batch_size)))
        for m in range(nb_batches):
            # batch indices
            begin, end = m * self.batch_size, min((m + 1) * self.batch_size, nb_samples)

            # create batch of adversarial examples
            x_imperceptible[begin:end] = self._generate_batch(x[begin:end], y[begin:end])
        return np.array(x_imperceptible, dtype=object)

    def _generate_batch(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Create imperceptible, adversarial sample.

        This is a helper method that calls the methods to create an adversarial (`ImperceptibleASR._create_adversarial`)
        and imperceptible (`ImperceptibleASR._create_imperceptible`) example subsequently.
        """
        # create adversarial example
        x_adversarial = self._create_adversarial(x, y)

        # make adversarial example imperceptible
        x_imperceptible = self._create_imperceptible(x, x_adversarial, y)
        return x_imperceptible

    def _create_adversarial(self, x, y) -> np.ndarray:
        """
        Create adversarial example with small perturbation that successfully deceives the estimator.

        The method implements the part of the paper by Qin et al. (2019) that is referred to as the first stage of the
        attack. The authors basically follow Carlini and Wagner (2018).

        | Paper link: https://arxiv.org/abs/1801.01944.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different
            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: An array with the adversarial outputs.
        """
        batch_size = x.shape[0]

        epsilon = [self.eps] * batch_size
        x_adversarial = [None] * batch_size

        x_perturbed = x.copy()

        for i in range(self.max_iter_1):
            # perform FGSM step for x
            gradients = self.estimator.loss_gradient(x_perturbed, y, batch_mode=True)
            x_perturbed = x_perturbed - self.learning_rate_1 * np.array([np.sign(g) for g in gradients], dtype=object)

            # clip perturbation
            perturbation = x_perturbed - x
            perturbation = np.array([np.clip(p, -e, e) for p, e in zip(perturbation, epsilon)], dtype=object)

            # re-apply clipped perturbation to x
            x_perturbed = x + perturbation

            if i % 10 == 0:
                prediction = self.estimator.predict(x_perturbed, batch_size=batch_size)
                for j in range(batch_size):
                    # validate adversarial target, i.e. f(x_perturbed)=y
                    if prediction[j] == y[j].upper():
                        # decrease max norm bound epsilon
                        perturbation_norm = np.max(np.abs(perturbation[j]))
                        if epsilon[j] > perturbation_norm:
                            epsilon[j] = perturbation_norm
                        epsilon[j] *= 0.8
                        # save current best adversarial example
                        x_adversarial[j] = x_perturbed[j]
                logger.info("Current iteration %s, epsilon %s", i, epsilon)

        # return perturbed x if no adversarial example found
        for j in range(batch_size):
            if x_adversarial[j] is None:
                logger.critical("Adversarial attack stage 1 for x_%s was not successful", j)
                x_adversarial[j] = x_perturbed[j]

        return np.array(x_adversarial, dtype=object)

    def _create_imperceptible(self, x: np.ndarray, x_adversarial: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Create imperceptible, adversarial example with small perturbation.

        This method implements the part of the paper by Qin et al. (2019) that is described as the second stage of the
        attack. The resulting adversarial audio samples are able to successfully deceive the ASR estimator and are
        imperceptible to the human ear.

        :param x: An array with the original inputs to be attacked.
        :param x_adversarial: An array with the adversarial examples.
        :param y: Target values of shape (batch_size,). Each sample in `y` is a string and it may possess different
            lengths. A possible example of `y` could be: `y = np.array(['SIXTY ONE', 'HELLO'])`.
        :return: An array with the imperceptible, adversarial outputs.
        """
        batch_size = x.shape[0]
        alpha_min = 0.0005

        alpha = np.array([self.alpha] * batch_size)
        loss_theta_previous = [np.inf] * batch_size
        x_imperceptible = [None] * batch_size
        # if inputs are *not* ragged, we can't multiply alpha * gradients_theta
        if x.ndim != 1:
            alpha = np.expand_dims(alpha, axis=-1)

        x_perturbed = x_adversarial.copy()

        for i in range(self.max_iter_2):
            # get perturbation
            perturbation = x_perturbed - x

            # get loss gradients of both losses
            gradients_net = self.estimator.loss_gradient(x_perturbed, y, batch_mode=True)
            gradients_theta, loss_theta = self._loss_gradient_masking_threshold(perturbation, x)

            # perform gradient descent steps
            x_perturbed = x_perturbed - self.learning_rate_2 * (gradients_net + alpha * gradients_theta)

            if i % 20 == 0 or i % 50 == 0:
                prediction = self.estimator.predict(x_perturbed, batch_size=batch_size)
                for j in range(batch_size):
                    # validate if adversarial target succeeds, i.e. f(x_perturbed)=y
                    if i % 20 == 0 and prediction[j] == y[j].upper():
                        # increase alpha
                        alpha[j] *= 1.2
                        # save current best imperceptible, adversarial example
                        if loss_theta[j] < loss_theta_previous[j]:
                            x_imperceptible[j] = x_perturbed[j]
                            loss_theta_previous[j] = loss_theta[j]

                    # validate if adversarial target fails, i.e. f(x_perturbed)!=y
                    if i % 50 == 0 and prediction[j] != y[j].upper():
                        # decrease alpha
                        alpha[j] = max(alpha[j] * 0.8, alpha_min)
                logger.info("Current iteration %s, alpha %s, loss theta %s", i, alpha, loss_theta)

        # return perturbed x if no adversarial example found
        for j in range(batch_size):
            if x_imperceptible[j] is None:
                logger.critical("Adversarial attack stage 2 for x_%s was not successful", j)
                x_imperceptible[j] = x_perturbed[j]

        return np.array(x_imperceptible, dtype=object)

    def _loss_gradient_masking_threshold(
        self, perturbation: np.ndarray, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute loss gradient of the global masking threshold w.r.t. the PSD approximate of the perturbation.

        The loss is defined as the hinge loss w.r.t. to the frequency masking threshold of the original audio input `x`
        and the normalized power spectral density estimate of the perturbation. In order to stabilize the optimization
        problem during back-propagation, the `10*log`-terms are canceled out.

        :param perturbation: Adversarial perturbation.
        :param x: An array with the original inputs to be attacked.
        :return: Tuple consisting of the loss gradient, which has same shape as `perturbation`, and loss value.
        """
        # pad input
        perturbation_padded, delta_mask = pad_sequence_input(perturbation)
        x_padded, _ = pad_sequence_input(x)

        # calculate masking threshold and PSD maximum
        masking_threshold = []
        psd_maximum = []
        for x_i in x_padded:
            mt, pm = self.masker.calculate_threshold_and_psd_maximum(x_i)
            masking_threshold.append(mt)
            psd_maximum.append(pm)
        masking_threshold = np.array(masking_threshold)
        psd_maximum = np.array(psd_maximum)

        # stabilize masking threshold loss by canceling out the "10*log" term in power spectral density and masking
        # threshold
        masking_threshold_stabilized = 10 ** (masking_threshold * 0.1)
        psd_maximum_stabilized = 10 ** (psd_maximum * 0.1)

        if self._framework == "tensorflow":
            # get loss gradients (TensorFlow)
            feed_dict = {
                self._delta: perturbation_padded,
                self._power_spectral_density_maximum_tf: psd_maximum_stabilized,
                self._masking_threshold_tf: masking_threshold_stabilized,
            }
            gradients_padded, loss = self.estimator._sess.run(self._loss_gradient_masking_threshold_op_tf, feed_dict)
        elif self._framework == "pytorch":
            # get loss gradients (TensorFlow)
            gradients_padded, loss = self._loss_gradient_masking_threshold_torch(
                perturbation_padded, psd_maximum_stabilized, masking_threshold_stabilized
            )
        else:
            raise NotImplementedError

        # undo padding, i.e. change gradients shape from (nb_samples, max_length) to (nb_samples)
        lengths = delta_mask.sum(axis=1)
        gradients = list()
        for gradient_padded, length in zip(gradients_padded, lengths):
            gradient = gradient_padded[:length]
            gradients.append(gradient)

        return np.array(gradients, dtype=object), loss

    def _loss_gradient_masking_threshold_tf(
        self, perturbation: "Tensor", psd_maximum_stabilized: "Tensor", masking_threshold_stabilized: "Tensor"
    ) -> Union["Tensor", "Tensor"]:
        """
        Compute loss gradient of the masking threshold loss in TensorFlow.

        Note that the PSD maximum and masking threshold are required to be stabilized, i.e. have the `10*log10`-term
        canceled out. Following Qin et al (2019) this mitigates optimization instabilities.

        :param perturbation: Adversarial perturbation.
        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of
            the original unnormalized PSD of `x`.
        :param masking_threshold_stabilized: Stabilized masking threshold for the original input `x`.
        :return: Approximate PSD tensor of shape `(batch_size, window_size // 2 + 1, frame_length)`.
        """
        import tensorflow.compat.v1 as tf1

        # calculate approximate power spectral density
        psd_perturbation = self._approximate_power_spectral_density_tf(perturbation, psd_maximum_stabilized)

        # calculate hinge loss
        loss = tf1.reduce_mean(
            tf1.nn.relu(psd_perturbation - masking_threshold_stabilized), axis=[1, 2], keepdims=False
        )

        # compute loss gradient
        loss_gradient = tf1.gradients(loss, [perturbation])[0]
        return loss_gradient, loss

    def _loss_gradient_masking_threshold_torch(
        self, perturbation: np.ndarray, psd_maximum_stabilized: np.ndarray, masking_threshold_stabilized: np.ndarray
    ) -> Union[np.ndarray, np.ndarray]:
        """
        Compute loss gradient of the masking threshold loss in PyTorch.

        See also `ImperceptibleASR._loss_gradient_masking_threshold_tf`.
        """
        import torch  # lgtm [py/import-and-import-from]

        # define tensors
        perturbation_torch = torch.from_numpy(perturbation).to(self.estimator._device)
        masking_threshold_stabilized_torch = torch.from_numpy(masking_threshold_stabilized).to(self.estimator._device)
        psd_maximum_stabilized_torch = torch.from_numpy(psd_maximum_stabilized).to(self.estimator._device)

        # track gradient of perturbation
        perturbation_torch.requires_grad = True

        # calculate approximate power spectral density
        psd_perturbation = self._approximate_power_spectral_density_torch(
            perturbation_torch, psd_maximum_stabilized_torch
        )

        # calculate hinge loss
        loss = torch.mean(
            torch.nn.functional.relu(psd_perturbation - masking_threshold_stabilized_torch), dim=(1, 2), keepdims=False
        )

        # compute loss gradient
        loss.sum().backward()
        loss_gradient = perturbation_torch.grad.cpu().numpy()
        loss_value = loss.detach().cpu().numpy()

        return loss_gradient, loss_value

    def _approximate_power_spectral_density_tf(
        self, perturbation: "Tensor", psd_maximum_stabilized: "Tensor"
    ) -> "Tensor":
        """
        Approximate the power spectral density for a perturbation `perturbation` in TensorFlow.

        Note that a stabilized PSD approximate is returned, where the `10*log10`-term has been canceled out.
        Following Qin et al (2019) this mitigates optimization instabilities.

        :param perturbation: Adversarial perturbation.
        :param psd_maximum_stabilized: Stabilized maximum across frames, i.e. shape is `(batch_size, frame_length)`, of
            the original unnormalized PSD of `x`.
        :return: Approximate PSD tensor of shape `(batch_size, window_size // 2 + 1, frame_length)`.
        """
        import tensorflow.compat.v1 as tf1

        # compute short-time Fourier transform (STFT)
        stft_matrix = tf1.signal.stft(perturbation, self._window_size, self._hop_size, fft_length=self._window_size)

        # compute power spectral density (PSD)
        # note: fixes implementation of Qin et al. by also considering the square root of gain_factor
        gain_factor = 8.0 / 3.0
        psd_matrix = gain_factor * tf1.square(tf1.abs(stft_matrix / self._window_size))

        # approximate normalized psd: psd_matrix_approximated = 10^((96.0 - psd_matrix_max + psd_matrix)/10)
        psd_matrix_approximated = tf1.pow(10.0, 9.6) / tf1.expand_dims(psd_maximum_stabilized, -1) * psd_matrix

        # return PSD matrix such that shape is (batch_size, window_size // 2 + 1, frame_length)
        return tf1.transpose(psd_matrix_approximated, [0, 2, 1])

    def _approximate_power_spectral_density_torch(
        self, perturbation: "PTensor", psd_maximum_stabilized: "PTensor"
    ) -> "PTensor":
        """
        Approximate the power spectral density for a perturbation `perturbation` in PyTorch.

        See also `ImperceptibleASR._approximate_power_spectral_density_tf`.
        """
        import torch  # lgtm [py/import-and-import-from]

        # compute short-time Fourier transform (STFT)
        stft_matrix = torch.stft(
            perturbation,
            n_fft=self._window_size,
            hop_length=self._hop_size,
            win_length=self._window_size,
            center=False,
            window=torch.hann_window(self._window_size).to(self.estimator._device),
        ).to(self.estimator._device)
        stft_matrix_abs = torch.sqrt(torch.sum(torch.square(stft_matrix), -1))

        # compute power spectral density (PSD)
        # note: fixes implementation of Qin et al. by also considering the square root of gain_factor
        gain_factor = 8.0 / 3.0
        psd_matrix = gain_factor * torch.square(stft_matrix_abs / self._window_size)

        # approximate normalized psd: psd_matrix_approximated = 10^((96.0 - psd_matrix_max + psd_matrix)/10)
        psd_matrix_approximated = pow(10.0, 9.6) / torch.unsqueeze(psd_maximum_stabilized, 1) * psd_matrix

        # return PSD matrix such that shape is (batch_size, window_size // 2 + 1, frame_length)
        return psd_matrix_approximated

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        if self.eps <= 0:
            raise ValueError("The perturbation max norm bound `eps` has to be positive.")

        if not isinstance(self.alpha, float):
            raise ValueError("The value of alpha must be of type float.")
        if self.alpha <= 0.0:
            raise ValueError("The value of alpha must be positive")

        if not isinstance(self.max_iter_1, int):
            raise ValueError("The maximum number of iterations for stage 1 must be of type int.")
        if self.max_iter_1 <= 0:
            raise ValueError("The maximum number of iterations for stage 1 must be greater than 0.")

        if not isinstance(self.max_iter_2, int):
            raise ValueError("The maximum number of iterations for stage 2 must be of type int.")
        if self.max_iter_2 < 0:
            raise ValueError("The maximum number of iterations for stage 2 must be non-negative.")

        if not isinstance(self.learning_rate_1, float):
            raise ValueError("The learning rate for stage 1 must be of type float.")
        if self.learning_rate_1 <= 0.0:
            raise ValueError("The learning rate for stage 1 must be greater than 0.0.")

        if not isinstance(self.learning_rate_2, float):
            raise ValueError("The learning rate for stage 2 must be of type float.")
        if self.learning_rate_2 <= 0.0:
            raise ValueError("The learning rate for stage 2 must be greater than 0.0.")

        if self.batch_size <= 0:
            raise ValueError("The batch size `batch_size` has to be positive.")


class PsychoacousticMasker:
    """
    Implements psychoacoustic model of Lin and Abdulla (2015) following Qin et al. (2019) simplifications.

    | Paper link: Lin and Abdulla (2015), https://www.springer.com/gp/book/9783319079738
    | Paper link: Qin et al. (2019), http://proceedings.mlr.press/v97/qin19a.html
    """

    def __init__(self, window_size: int = 2048, hop_size: int = 512, sample_rate: int = 16000) -> None:
        """
        Initialization.

        :param window_size: Length of the window. The number of STFT rows is `(window_size // 2 + 1)`.
        :param hop_size: Number of audio samples between adjacent STFT columns.
        :param sample_rate: Sampling frequency of audio inputs.
        """
        self._window_size = window_size
        self._hop_size = hop_size
        self._sample_rate = sample_rate

        # init some private properties for lazy loading
        self._fft_frequencies = None
        self._bark = None
        self._absolute_threshold_hearing = None

    def calculate_threshold_and_psd_maximum(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the global masking threshold for an audio input and also return its maximum power spectral density.

        This method is the main method to call in order to obtain global masking thresholds for an audio input. It also
        returns the maximum power spectral density (PSD) for each frame. Given an audio input, the following steps are
        performed:

        1. STFT analysis and sound pressure level normalization
        2. Identification and filtering of maskers
        3. Calculation of individual masking thresholds
        4. Calculation of global masking tresholds

        :param audio: Audio samples of shape `(length,)`.
        :return: Global masking thresholds of shape `(window_size // 2 + 1, frame_length)` and the PSD maximum for each
            frame of shape `(frame_length)`.
        """
        psd_matrix, psd_max = self.power_spectral_density(audio)
        threshold = np.zeros(psd_matrix.shape)
        for frame in range(psd_matrix.shape[1]):
            # apply methods for finding and filtering maskers
            maskers, masker_idx = self.filter_maskers(*self.find_maskers(psd_matrix[:, frame]))
            # apply methods for calculating global threshold
            threshold[:, frame] = self.calculate_global_threshold(
                self.calculate_individual_threshold(maskers, masker_idx)
            )
        return threshold, psd_max

    @property
    def window_size(self) -> int:
        """
        :return: Window size of the masker.
        """
        return self._window_size

    @property
    def hop_size(self) -> int:
        """
        :return: Hop size of the masker.
        """
        return self._hop_size

    @property
    def sample_rate(self) -> int:
        """
        :return: Sample rate of the masker.
        """
        return self._sample_rate

    @property
    def fft_frequencies(self) -> np.ndarray:
        """
        :return: Discrete fourier transform sample frequencies.
        """
        if self._fft_frequencies is None:
            self._fft_frequencies = np.linspace(0, self.sample_rate / 2, self.window_size // 2 + 1)
        return self._fft_frequencies

    @property
    def bark(self) -> np.ndarray:
        """
        :return: Bark scale for discrete fourier transform sample frequencies.
        """
        if self._bark is None:
            self._bark = 13 * np.arctan(0.00076 * self.fft_frequencies) + 3.5 * np.arctan(
                np.square(self.fft_frequencies / 7500.0)
            )
        return self._bark

    @property
    def absolute_threshold_hearing(self) -> np.ndarray:
        """
        :return: Absolute threshold of hearing (ATH) for discrete fourier transform sample frequencies.
        """
        if self._absolute_threshold_hearing is None:
            # ATH applies only to frequency range 20Hz<=f<=20kHz
            # note: deviates from Qin et al. implementation by using the Hz range as valid domain
            valid_domain = np.logical_and(20 <= self.fft_frequencies, self.fft_frequencies <= 2e4)
            freq = self.fft_frequencies[valid_domain] * 0.001

            # outside valid ATH domain, set values to np.inf
            self._absolute_threshold_hearing = np.ones(valid_domain.shape) * np.inf
            self._absolute_threshold_hearing[valid_domain] = (
                3.64 * pow(freq, -0.8) - 6.5 * np.exp(-0.6 * np.square(freq - 3.3)) + 0.001 * pow(freq, 4) - 12
            )
        return self._absolute_threshold_hearing

    def power_spectral_density(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the power spectral density matrix for an audio input.

        :param audio: Audio sample of shape `(length,)`.
        :return: PSD matrix of shape `(window_size // 2 + 1, frame_length)` and maximum vector of shape
        `(frame_length)`.
        """
        # compute short-time Fourier transform (STFT)
        stft_params = {
            "fs": self.sample_rate,
            "window": ss.get_window("hann", self.window_size, fftbins=True),
            "nperseg": self.window_size,
            "nfft": self.window_size,
            "noverlap": self.window_size - self.hop_size,
            "boundary": None,
            "padded": False,
        }
        _, _, stft_matrix = ss.stft(audio, **stft_params)

        # undo SciPy's hard-coded normalization
        # https://github.com/scipy/scipy/blob/01d8bfb6f239df4ce70c799b9b485b53733c9911/scipy/signal/spectral.py#L1802
        stft_matrix *= stft_params["window"].sum()

        # compute power spectral density (PSD)
        gain_factor = np.sqrt(8.0 / 3.0)
        psd_matrix = 10 * np.log10(np.abs(gain_factor * stft_matrix / self.window_size) ** 2 + np.finfo(np.float32).eps)

        # normalize PSD at 96dB
        # note: deviates from Qin et al. implementation by taking maximum across frames
        psd_matrix_max = np.max(psd_matrix, axis=0)
        psd_matrix_normalized = 96.0 - psd_matrix_max + psd_matrix

        return psd_matrix_normalized, psd_matrix_max

    @staticmethod
    def find_maskers(psd_vector: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify maskers.

        Possible maskers are local PSD maxima. Following Qin et al., all maskers are treated as tonal. Thus neglecting
        the nontonal type.

        :param psd_vector: PSD vector of shape `(window_size // 2 + 1)`.
        :return: Possible PSD maskers and indices.
        """
        # identify maskers. For simplification it is assumed that all maskers are tonal (vs. nontonal).
        masker_idx = ss.argrelmax(psd_vector)[0]

        # smooth maskers with their direct neighbors
        psd_maskers = 10 * np.log10(np.sum([10 ** (psd_vector[masker_idx + i] / 10) for i in range(-1, 2)], axis=0))
        return psd_maskers, masker_idx

    def filter_maskers(self, maskers: np.ndarray, masker_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter maskers.

        First, discard all maskers that are below the absolute threshold of hearing. Second, reduce pairs of maskers
        that are within 0.5 bark distance of each other by keeping the larger masker.

        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Filtered PSD maskers and indices.
        """
        # filter on the absolute threshold of hearing
        # note: deviates from Qin et al. implementation by filtering first on ATH and only then on bark distance
        ath_condition = maskers > self.absolute_threshold_hearing[masker_idx]
        masker_idx = masker_idx[ath_condition]
        maskers = maskers[ath_condition]

        # filter on the bark distance
        bark_condition = np.ones(masker_idx.shape, dtype=bool)
        i_prev = 0
        for i in range(1, len(masker_idx)):
            # find pairs of maskers that are within 0.5 bark distance of each other
            if self.bark[i] - self.bark[i_prev] < 0.5:
                # discard the smaller masker
                i_todelete, i_prev = (i_prev, i_prev + 1) if maskers[i_prev] < maskers[i] else (i, i_prev)
                bark_condition[i_todelete] = False
            else:
                i_prev = i
        masker_idx = masker_idx[bark_condition]
        maskers = maskers[bark_condition]

        return maskers, masker_idx

    def calculate_individual_threshold(self, maskers: np.ndarray, masker_idx: np.ndarray) -> np.ndarray:
        """
        Calculate individual masking threshold with frequency denoted at bark scale.

        :param maskers: Masker PSD values.
        :param masker_idx: Masker indices.
        :return: Individual threshold vector of shape `(window_size // 2 + 1)`.
        """
        delta_shift = -6.025 - 0.275 * self.bark
        threshold = np.zeros(masker_idx.shape + self.bark.shape)
        # TODO reduce for loop
        for k, (masker_j, masker) in enumerate(zip(masker_idx, maskers)):
            # critical band rate of the masker
            z_j = self.bark[masker_j]
            # distance maskees to masker in bark
            delta_z = self.bark - z_j
            # define two-slope spread function:
            #   if delta_z <= 0, spread_function = 27*delta_z
            #   if delta_z > 0, spread_function = [-27+0.37*max(PSD_masker-40,0]*delta_z
            spread_function = 27 * delta_z
            spread_function[delta_z > 0] = (-27 + 0.37 * max(masker - 40, 0)) * delta_z[delta_z > 0]

            # calculate threshold
            threshold[k, :] = masker + delta_shift[masker_j] + spread_function
        return threshold

    def calculate_global_threshold(self, individual_threshold):
        """
        Calculate global masking threshold.

        :param individual_threshold: Individual masking threshold vector.
        :return: Global threshold vector of shape `(window_size // 2 + 1)`.
        """
        # note: deviates from Qin et al. implementation by taking the log of the summation, which they do for numerical
        #       stability of the stage 2 optimization. We stabilize the optimization in the loss itself.
        return 10 * np.log10(
            np.sum(10 ** (individual_threshold / 10), axis=0) + 10 ** (self.absolute_threshold_hearing / 10)
        )
