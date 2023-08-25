"""
This module contains the Huggingface API.
"""
import importlib

if importlib.util.find_spec("torch") is not None:
    from art.estimators.hugging_face.hugging_face import HuggingFaceClassifierPyTorch
