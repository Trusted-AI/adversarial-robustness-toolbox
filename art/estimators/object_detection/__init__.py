"""
Module containing estimators for object detection.
"""
from art.estimators.object_detection.object_detector import ObjectDetectorMixin

from art.estimators.object_detection.pytorch_object_detector import PyTorchObjectDetector
from art.estimators.object_detection.pytorch_faster_rcnn import PyTorchFasterRCNN
from art.estimators.object_detection.pytorch_yolo import PyTorchYolo
from art.estimators.object_detection.tensorflow_faster_rcnn import TensorFlowFasterRCNN
from art.estimators.object_detection.tensorflow_v2_faster_rcnn import TensorFlowV2FasterRCNN
