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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import pytest
import os

from art.attacks.poisoning.perturbations.audio_perturbations import CacheToneTrigger, CacheAudioTrigger

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.framework_agnostic
def test_insert_tone_trigger(art_warning):
    try:
        # test single example
        trigger = CacheToneTrigger(sampling_rate=16000)
        audio = trigger.insert(x=np.zeros(3200))
        assert audio.shape == (3200,)
        assert np.max(audio) != 0
        assert np.max(np.abs(audio)) <= 1.0

        # test single example with differet duration, frequency, and scale
        trigger = CacheToneTrigger(sampling_rate=16000, frequency=16000, duration=0.2, scale=0.5)
        audio = trigger.insert(x=np.zeros(3200))
        assert audio.shape == (3200,)
        assert np.max(audio) != 0

        # test a batch of examples
        audio = trigger.insert(x=np.zeros((10, 3200)))
        assert audio.shape == (10, 3200)
        assert np.max(audio) != 0

        # test single example with shift
        trigger = CacheToneTrigger(sampling_rate=16000, shift=10)
        audio = trigger.insert(x=np.zeros(3200))
        assert audio.shape == (3200,)
        assert np.max(audio) != 0
        assert np.sum(audio[:10]) == 0

        # test a batch of examples with random shift
        trigger = CacheToneTrigger(sampling_rate=16000, random=True)
        audio = trigger.insert(x=np.zeros((10, 3200)))
        assert audio.shape == (10, 3200)
        assert np.max(audio) != 0

        # test when length of backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            trigger = CacheToneTrigger(sampling_rate=16000, duration=0.3)
            _ = trigger.insert(x=np.zeros(3200))

        # test when shift + backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            trigger = CacheToneTrigger(sampling_rate=16000, duration=0.2, shift=5)
            _ = trigger.insert(x=np.zeros(3200))

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.framework_agnostic
def test_insert_audio_trigger(art_warning):
    file_path = os.path.join(os.getcwd(), "utils/data/backdoors/cough_trigger.wav")
    try:
        # test single example
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path)
        audio = trigger.insert(x=np.zeros(32000))
        assert audio.shape == (32000,)
        assert np.max(audio) != 0
        assert np.max(np.abs(audio)) <= 1.0

        # test single example with differet duration and scale
        trigger = CacheAudioTrigger(
            sampling_rate=16000,
            backdoor_path=file_path,
            duration=0.8,
            scale=0.5,
        )
        audio = trigger.insert(x=np.zeros(32000))
        assert audio.shape == (32000,)
        assert np.max(audio) != 0

        # test a batch of examples
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path)
        audio = trigger.insert(x=np.zeros((10, 16000)))

        assert audio.shape == (10, 16000)
        assert np.max(audio) != 0

        # test single example with shift
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path, shift=10)
        audio = trigger.insert(x=np.zeros(32000))
        assert audio.shape == (32000,)
        assert np.max(audio) != 0
        assert np.sum(audio[:10]) == 0

        # test a batch of examples with random shift
        trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path, random=True)
        audio = trigger.insert(x=np.zeros((10, 32000)))
        assert audio.shape == (10, 32000)
        assert np.max(audio) != 0

        # test when length of backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            trigger = CacheAudioTrigger(sampling_rate=16000, backdoor_path=file_path)
            _ = trigger.insert(x=np.zeros(15000))

        # test when shift + backdoor is larger than that of audio signal
        with pytest.raises(ValueError):
            trigger = CacheAudioTrigger(
                sampling_rate=16000,
                backdoor_path=file_path,
                duration=1,
                shift=5,
            )
            _ = trigger.insert(x=np.zeros(16000))

    except ARTTestException as e:
        art_warning(e)
