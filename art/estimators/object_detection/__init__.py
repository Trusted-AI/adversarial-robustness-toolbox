"""
Module containing estimators for object detection.
"""
from art.estimators.object_detection.object_detector import ObjectDetectorMixin

from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
