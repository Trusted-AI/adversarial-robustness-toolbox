"""
Module containing estimators for speech recognition.
"""
from art.estimators.speech_recognition.speech_recognizer import SpeechRecognizerMixin

from art.estimators.speech_recognition.pytorch_deep_speech import PyTorchDeepSpeech
from art.estimators.speech_recognition.pytorch_espresso import PyTorchEspresso
from art.estimators.speech_recognition.tensorflow_lingvo import TensorFlowLingvoASR
