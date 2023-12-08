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
This module implements the abstract estimator `HuggingFaceClassifier` using the PyTorchClassifier as a backend
to interface with ART.
"""
import logging

from typing import List, Optional, Tuple, Union, Dict, Callable, Any, TYPE_CHECKING

import numpy as np
import six

from art.estimators.language_modeling.language_model import LanguageModel

if TYPE_CHECKING:
    import torch
    import transformers
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
    from transformers.modeling_outputs import ModelOutput

logger = logging.getLogger(__name__)


class HuggingFaceLM(LanguageModel):
    """
    This class implements a classifier with the HuggingFace framework.
    """
    
    pass
