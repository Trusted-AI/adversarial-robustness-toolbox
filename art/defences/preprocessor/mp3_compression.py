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
This module implements the MP3 compression defence `Mp3Compression`.

| Paper link: https://arxiv.org/abs/1801.01944

| Please keep in mind the limitations of defences. For details on how to evaluate classifier security in general,
    see https://arxiv.org/abs/1902.06705.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

from art.defences.preprocessor.preprocessor import Preprocessor

logger = logging.getLogger(__name__)


class Mp3Compression(Preprocessor):
    """
    Implement the MP3 compression defense approach.
    """

    params = ["channels_first", "sample_rate", "verbose"]

    def __init__(
        self,
        sample_rate: int,
        channels_first: bool = False,
        apply_fit: bool = False,
        apply_predict: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Create an instance of MP3 compression.

        :param sample_rate: Specifies the sampling rate of sample.
        :param channels_first: Set channels first or last.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        :param verbose: Show progress bars.
        """
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.channels_first = channels_first
        self.sample_rate = sample_rate
        self.verbose = verbose
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply MP3 compression to sample `x`.

        :param x: Sample to compress with shape `(batch_size, length, channel)` or an array of sample arrays with shape
                  (length,) or (length, channel).
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :return: Compressed sample.
        """

        def wav_to_mp3(x, sample_rate):
            """
            Apply MP3 compression to audio input of shape (samples, channel).
            """
            from pydub import AudioSegment
            from scipy.io.wavfile import write

            x_dtype = x.dtype
            normalized = bool(x.min() >= -1.0 and x.max() <= 1.0)
            if x_dtype != np.int16 and not normalized:
                # input is not of type np.int16 and seems to be unnormalized. Therefore casting to np.int16.
                x = x.astype(np.int16)
            elif x_dtype != np.int16 and normalized:
                # x is not of type np.int16 and seems to be normalized. Therefore undoing normalization and
                # casting to np.int16.
                x = (x * 2 ** 15).astype(np.int16)

            tmp_wav, tmp_mp3 = BytesIO(), BytesIO()
            write(tmp_wav, sample_rate, x)
            AudioSegment.from_wav(tmp_wav).export(tmp_mp3)
            audio_segment = AudioSegment.from_mp3(tmp_mp3)
            tmp_wav.close()
            tmp_mp3.close()
            x_mp3 = np.array(audio_segment.get_array_of_samples()).reshape((-1, audio_segment.channels))

            # WARNING: Sometimes we *still* need to manually resize x_mp3 to original length.
            # This should not be the case, e.g. see https://github.com/jiaaro/pydub/issues/474
            if x.shape[0] != x_mp3.shape[0]:
                logger.warning(
                    "Lengths original input and compressed output don't match. Truncating compressed result."
                )
                x_mp3 = x_mp3[: x.shape[0]]

            if normalized:
                # x was normalized. Therefore normalizing x_mp3.
                x_mp3 = x_mp3 * 2 ** -15
            return x_mp3.astype(x_dtype)

        if x.dtype != np.object and x.ndim != 3:
            raise ValueError("Mp3 compression can only be applied to temporal data across at least one channel.")

        if x.dtype != np.object and self.channels_first:
            x = np.swapaxes(x, 1, 2)

        # apply mp3 compression per audio item
        x_mp3 = x.copy()
        for i, x_i in enumerate(tqdm(x, desc="MP3 compression", disable=not self.verbose)):
            x_i_ndim_0 = x_i.ndim
            if x.dtype == np.object:
                if x_i.ndim == 1:
                    x_i = np.expand_dims(x_i, axis=1)

                if x_i_ndim_0 == 2 and self.channels_first:
                    x_i = np.swapaxes(x_i, 0, 1)

            x_i = wav_to_mp3(x_i, self.sample_rate)

            if x.dtype == np.object:
                if x_i_ndim_0 == 2 and self.channels_first:
                    x_i = np.swapaxes(x_i, 0, 1)

                if x_i_ndim_0 == 1:
                    x_i = np.squeeze(x_i)

            x_mp3[i] = x_i

        if x.dtype != np.object and self.channels_first:
            x_mp3 = np.swapaxes(x_mp3, 1, 2)

        return x_mp3, y

    def _check_params(self) -> None:
        if not (isinstance(self.sample_rate, (int, np.int)) and self.sample_rate > 0):
            raise ValueError("Sample rate be must a positive integer.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")
