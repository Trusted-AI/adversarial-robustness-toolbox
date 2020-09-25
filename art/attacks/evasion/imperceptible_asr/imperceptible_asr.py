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
from typing import Tuple

import numpy as np
import scipy.signal as ss

from art.attacks.attack import EvasionAttack
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin
from art.estimators.tensorflow import TensorFlowV2Estimator

logger = logging.getLogger(__name__)


class ImperceptibleAsr(EvasionAttack):
    """
    Implementation of the imperceptible attack against a speech recognition model.

    | Paper link: http://proceedings.mlr.press/v97/qin19a.html
    """

    attack_params = EvasionAttack.attack_params + [
        "eps",
        "learning_rate_1",
        "max_iter_1",
    ]

    _estimator_requirements = (TensorFlowV2Estimator, SpeechRecognizerMixin)

    def __init__(
        self,
        estimator: "TensorFlowV2Estimator",
        eps: float = 2000,
        learning_rate_1: float = 100,
        max_iter_1: int = 1000,
    ) -> None:
        """
        Create an instance of the :class:`.ImperceptibleAsr`.

        :param estimator: A trained classifier.
        :param eps: Initial max norm bound for adversarial perturbation.
        :param learning_rate_1: Learning rate for stage 1 of attack.
        :param max_iter_1: Number of iterations for stage 1 of attack.
        """
        # Super initialization
        super().__init__(estimator=estimator)
        self.eps = eps
        self.learning_rate_1 = learning_rate_1
        self.max_iter_1 = max_iter_1
        self._targeted = True
        self._check_params()

    def generate(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array. This method should be overridden by all concrete
        evasion attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted or not. This parameter
            is only used by some of the attacks.
        :return: An array holding the adversarial examples.
        """
        pass

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
        perturbation = np.zeros_like(x_perturbed)

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

    def _check_params(self) -> None:
        """
        Apply attack-specific checks.
        """
        pass


class PsychoacousticMasker:
    """
    Implements psychoacoustic model of Lin and Abdulla (2015) following Qin et al. (2019) simplifications.

    | Lin and Abdulla (2015), https://www.springer.com/gp/book/9783319079738
    | Qin et al. (2019), http://proceedings.mlr.press/v97/qin19a.html
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

    def calculate_threshold_and_psd_maximum(self, audio: np.ndarray):
        """Compute the global masking threshold for an audio input and also return its maxium power spectral density.

        This method is the main method to call in order to obtain global masking thresholds for an audio input. It also
        returns the maxium power spectral density (PSD) for each frame. Given an audio input, the following steps are
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
        """Compute the power spectral density matrix for an audio input.

        :param audio: Audio sample of shape `(length,)`.
        :return: PSD matrix of shape `(window_size // 2 + 1, frame_length)` and maxium vector of shape `(frame_length)`.
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
        """Identify maskers.

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
        """Filter maskers.

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
        """Calculate individual masking threshold with frequency denoted at bark scale.

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
        """Calculate global masking threshold.

        :param individual_threshold: Individual masking threshold vector.
        :return: Global threshold vector of shape `(window_size // 2 + 1)`.
        """
        # note: deviates from Qin et al. implementation by taking the log of the summation, which they do for numerical
        #       stability of the stage 2 optimization. We stabilize the optimization in the loss itself.
        return 10 * np.log10(
            np.sum(10 ** (individual_threshold / 10), axis=0) + 10 ** (self.absolute_threshold_hearing / 10)
        )
