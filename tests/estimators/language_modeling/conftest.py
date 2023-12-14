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

logger = logging.getLogger(__name__)


@pytest.fixture()
def get_text_data():
    """
    Sample text data to test language model estimators
    """
    text = [
        "The capital of France is",
        "The capital of Germany is",
    ]
    tokens = np.asarray([[464, 3139, 286, 4881, 318], [464, 3139, 286, 4486, 318]])
    return text, tokens


@pytest.fixture()
def get_hugging_face_language_model(get_text_data):
    """
    This class tests the HuggingFaceLanguageModel estimator.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from art.estimators.language_modeling import HuggingFaceLanguageModel

    # Define language model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    language_model = HuggingFaceLanguageModel(
        model=model,
        tokenizer=tokenizer,
    )

    text, tokens = get_text_data

    yield language_model, text, tokens
