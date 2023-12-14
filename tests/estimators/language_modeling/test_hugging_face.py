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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import numpy as np
import pytest

from tests.utils import ARTTestException

logger = logging.getLogger(__name__)


@pytest.mark.only_with_platform("huggingface")
def test_tokenize(art_warning, get_hugging_face_language_model):
    try:
        language_model, text, _ = get_hugging_face_language_model

        result = language_model.tokenize(x=text, padding=True, return_tensors="np")

        assert list(result.keys()) == ["input_ids", "attention_mask"]

        assert result["input_ids"].shape == (2, 5)
        expected_input_ids = np.asarray([[464, 3139, 286, 4881, 318], [464, 3139, 286, 4486, 318]])
        np.testing.assert_array_equal(result["input_ids"], expected_input_ids)

        assert result["attention_mask"].shape == (2, 5)
        expected_attention_masks = np.asarray([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        np.testing.assert_array_equal(result["attention_mask"], expected_attention_masks)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("huggingface")
def test_encode(art_warning, get_hugging_face_language_model):
    try:
        language_model, text, tokens = get_hugging_face_language_model

        result = language_model.encode(x=text[0])
        np.testing.assert_array_equal(result, tokens[0])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("huggingface")
def test_batch_encode(art_warning, get_hugging_face_language_model):
    try:
        language_model, text, tokens = get_hugging_face_language_model

        result = language_model.batch_encode(x=text)
        np.testing.assert_array_equal(result, tokens)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("huggingface")
def test_decode(art_warning, get_hugging_face_language_model):
    try:
        language_model, text, tokens = get_hugging_face_language_model

        result = language_model.decode(x=tokens[0])
        np.testing.assert_array_equal(result, text[0])

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("huggingface")
def test_batch_decode(art_warning, get_hugging_face_language_model):
    try:
        language_model, text, tokens = get_hugging_face_language_model

        result = language_model.batch_decode(x=tokens)
        np.testing.assert_array_equal(result, text)

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("huggingface")
def test_predict(art_warning, get_hugging_face_language_model):
    try:
        language_model, text, _ = get_hugging_face_language_model

        result = language_model.predict(x=text)

        assert list(result.keys()) == ["logits", "past_key_values"]

        assert result["logits"].shape == (2, 5, 50257)
        expected_logits = np.asarray(
            [
                -36.287148,
                -35.011105,
                -38.079082,
                -37.782597,
                -36.693443,
                -37.765358,
                -35.628418,
                -36.034225,
                -34.752907,
                -37.38435,
            ]
        )
        np.testing.assert_array_almost_equal(result["logits"][0, 0, :10], expected_logits, decimal=2)

        assert result["past_key_values"][0][0].shape == (2, 12, 5, 64)
        expected_past_key_values = np.asarray(
            [
                -0.9420212,
                1.9022521,
                0.87219113,
                -0.15018058,
                0.4039965,
                0.20760098,
                -0.1169314,
                0.6232018,
                -1.7421292,
                0.63497376,
            ]
        )
        np.testing.assert_array_almost_equal(
            result["past_key_values"][0][0][0, 0, 0, :10], expected_past_key_values, decimal=2
        )

    except ARTTestException as e:
        art_warning(e)


@pytest.mark.only_with_platform("huggingface")
def test_generate(art_warning, get_hugging_face_language_model):
    try:
        language_model, text, _ = get_hugging_face_language_model

        result = language_model.generate(x=text, max_new_tokens=1)

        assert len(result[0]) > len(text[0])
        assert len(result[1]) > len(text[1])

    except ARTTestException as e:
        art_warning(e)
