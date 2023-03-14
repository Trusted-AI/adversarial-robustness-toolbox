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
Adversarial perturbations designed to work for audio.
Uses classes, rather than pure functions as in image_perturbations.py,
because loading the audio trigger from disk (librosa.load()) is very slow
and should be done only once.
"""
import numpy as np
import librosa


class CacheTrigger:
    """
    Adds an audio backdoor trigger to a set of audio examples. Works for a single example or a batch of examples.
    """

    def __init__(
        self,
        trigger: np.ndarray,
        random: bool = False,
        shift: int = 0,
        scale: float = 0.1,
    ):
        """
        Initialize a CacheTrigger instance.

        :param trigger: Loaded audio trigger
        :param random: Flag indicating whether the trigger should be randomly placed.
        :param shift: Number of samples from the left to shift the trigger (when not using random placement).
        :param scale: Scaling factor for mixing the trigger.
        """
        self.trigger = trigger
        self.scaled_trigger = self.trigger * scale
        self.random = random
        self.shift = shift
        self.scale = scale

    def insert(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: N x L matrix or length L array, where N is number of examples, L is the length in number of samples.
                  X is in range [-1,1].
        :return: Backdoored audio.
        """
        n_dim = len(x.shape)
        if n_dim == 2:
            return np.array([self.insert(single_audio) for single_audio in x])
        if n_dim != 1:
            raise ValueError("Invalid array shape " + str(x.shape))
        original_dtype = x.dtype
        audio = np.copy(x)
        length = audio.shape[0]
        bd_length = self.trigger.shape[0]
        if bd_length > length:
            raise ValueError("Backdoor audio does not fit inside the original audio.")
        if self.random:
            shift = np.random.randint(length - bd_length)
        else:
            shift = self.shift
        if shift + bd_length > length:
            raise ValueError("Shift + Backdoor length is greater than audio's length.")

        audio[shift : shift + bd_length] += self.scaled_trigger
        audio = np.clip(audio, -1.0, 1.0)
        return audio.astype(original_dtype)


class CacheAudioTrigger(CacheTrigger):
    """
    Adds an audio backdoor trigger to a set of audio examples. Works for a single example or a batch of examples.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        backdoor_path: str = "../../../utils/data/backdoors/cough_trigger.wav",
        duration: float = None,
        **kwargs,
    ):
        """
        Initialize a CacheAudioTrigger instance.

        :param sampling_rate: Positive integer denoting the sampling rate for x.
        :param backdoor_path: The path to the audio to insert as a trigger.
        :param duration: Duration of the trigger in seconds. Default `None` if full trigger is to be used.
        """
        trigger, bd_sampling_rate = librosa.load(backdoor_path, mono=True, sr=None, duration=duration)

        if sampling_rate != bd_sampling_rate:
            print(
                f"Backdoor sampling rate {bd_sampling_rate} does not match with the sampling rate provided."
                "Resampling the backdoor to match the sampling rate."
            )
            trigger, _ = librosa.load(backdoor_path, mono=True, sr=sampling_rate, duration=duration)
        super().__init__(trigger, **kwargs)


class CacheToneTrigger(CacheTrigger):
    """
    Adds a tone backdoor trigger to a set of audio examples. Works for a single example or a batch of examples.
    """

    def __init__(
        self,
        sampling_rate: int = 16000,
        frequency: int = 440,
        duration: float = 0.1,
        **kwargs,
    ):
        """
        Initialize a CacheToneTrigger instance.

        :param sampling_rate: Positive integer denoting the sampling rate for x.
        :param frequency: Frequency of the tone to be added.
        :param duration: Duration of the tone to be added.
        """
        trigger = librosa.tone(frequency, sr=sampling_rate, duration=duration)
        super().__init__(trigger, **kwargs)
