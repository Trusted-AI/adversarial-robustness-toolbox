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
This module implements the task specific estimator for Faster R-CNN v3 in PyTorch.
"""
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING


from art.estimators.object_detection.python_object_detector import PyTorchObjectDetector

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torchvision

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchFasterRCNN(PyTorchObjectDetector):
    """
    This class implements a model-specific object detector using Faster-RCNN and PyTorch.
    """

    def __init__(
        self,
        model: Optional["torchvision.models.detection.fasterrcnn_resnet50_fpn"] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: Faster-RCNN model. The output of the model is `List[Dict[Tensor]]`, one for each input image. The
                      fields of the Dict are as follows:

                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                        between 0 and H and 0 and W
                      - labels (Int64Tensor[N]): the predicted labels for each image
                      - scores (Tensor[N]): the scores or each prediction
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param attack_losses: Tuple of any combination of strings of loss components: 'loss_classifier', 'loss_box_reg',
                              'loss_objectness', and 'loss_rpn_box_reg'.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torchvision  # lgtm [py/repeated-import]

        if model is None:  # pragma: no cover
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True, progress=True, num_classes=91, pretrained_backbone=True
            )

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            attack_losses=attack_losses,
            device_type=device_type,
        )
