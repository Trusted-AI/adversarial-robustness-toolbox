# MIT License
#
# Copyright (C) The Adversarial Robustness Toolbox (ART) Authors 2021
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
#
# MIT License
#
# Copyright (c) 2019 Nrupatunga
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the task specific estimator for PyTorch GOTURN object tracker.
"""
import logging
import time
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_tracking.object_tracker import ObjectTrackerMixin
from art.estimators.pytorch import PyTorchEstimator

if TYPE_CHECKING:
    # pylint: disable=C0412
    import PIL  # lgtm [py/import-and-import-from]
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchGoturn(ObjectTrackerMixin, PyTorchEstimator):
    """
    This module implements the task- and model-specific estimator for PyTorch GOTURN (object tracking).
    """

    estimator_params = PyTorchEstimator.estimator_params + ["attack_losses"]

    def __init__(
        self,
        model,
        input_shape: Tuple[int, ...],
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: GOTURN model.
        :param input_shape: Shape of one input sample as expected by the model, e.g. input_shape=(3, 227, 227).
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
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch  # lgtm [py/repeated-import]

        # Set device
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        model.to(self._device)

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )
        # got-10k toolkit
        # Tracker.__init__(self, name="PyTorchGoturn", is_deterministic=True)
        self.name = "PyTorchGoturn"
        self.is_deterministic = True

        self._input_shape = input_shape

        if self.clip_values is not None:
            if self.clip_values[0] != 0:  # pragma: no cover
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, 255).")
            if self.clip_values[1] not in [1, 255]:  # pragma: no cover
                raise ValueError(
                    "This classifier requires un-normalized input images with clip_vales=(0, 1) or "
                    "clip_vales=(0, 255)."
                )

        if self.postprocessing_defences is not None:  # pragma: no cover
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        self.attack_losses: Tuple[str, ...] = ("torch.nn.L1Loss",)

    @property
    def native_label_is_pytorch_format(self) -> bool:
        """
        Are the native labels in PyTorch format [x1, y1, x2, y2]?
        """
        return True

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """
        Return the shape of one input sample.

        :return: Shape of one input sample.
        """
        return self._input_shape  # type: ignore

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

    def _get_losses(
        self,
        x: np.ndarray,
        y: Union[List[Dict[str, np.ndarray]], List[Dict[str, "torch.Tensor"]]],
        reduction: str = "sum",
    ) -> Tuple[Dict[str, Union["torch.Tensor", int, List["torch.Tensor"]]], List["torch.Tensor"], List["torch.Tensor"]]:
        """
        Get the loss tensor output of the model including all preprocessing.

        :param x: Samples of shape (nb_samples, nb_frames, height, width, nb_channels).
        :param y: Target values of format `List[Dict[str, np.ndarray]]`, one dictionary for each input image. The keys
                  of the dictionary are:

                  - boxes [N_FRAMES, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                         0 <= y1 < y2 <= H.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'sum'.
                          'none': no reduction will be applied.
                          'sum': the output will be summed.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]

        self._model.train()

        # Apply preprocessing
        if self.all_framework_preprocessing:
            if isinstance(x, torch.Tensor):
                raise NotImplementedError

            if y is not None and isinstance(y[0]["boxes"], np.ndarray):
                y_tensor = list()
                for i, y_i in enumerate(y):
                    y_t = dict()
                    y_t["boxes"] = torch.from_numpy(y_i["boxes"]).float().to(self.device)
                    y_tensor.append(y_t)
            else:
                y_tensor = y

            image_tensor_list_grad = list()
            y_preprocessed = list()
            inputs_t: List["torch.Tensor"] = list()

            for i in range(x.shape[0]):
                if self.clip_values is not None:
                    x_grad = torch.from_numpy(x[i]).to(self.device).float()
                else:
                    x_grad = torch.from_numpy(x[i]).to(self.device).float()
                x_grad.requires_grad = True
                image_tensor_list_grad.append(x_grad)
                x_grad_1 = torch.unsqueeze(x_grad, dim=0)
                x_preprocessed_i, y_preprocessed_i = self._apply_preprocessing(
                    x_grad_1, y=[y_tensor[i]], fit=False, no_grad=False
                )
                x_preprocessed_i = torch.squeeze(x_preprocessed_i)
                y_preprocessed.append(y_preprocessed_i[0])
                inputs_t.append(x_preprocessed_i)

        elif isinstance(x, np.ndarray):
            raise NotImplementedError
        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        labels_t = y_preprocessed  # type: ignore

        if isinstance(y[0]["boxes"], np.ndarray):
            y_init = torch.from_numpy(y[0]["boxes"]).to(self.device)
        else:
            y_init = y[0]["boxes"]

        loss_list = list()

        for i in range(x.shape[0]):
            x_i = inputs_t[i]
            y_pred = self._track(x=x_i, y_init=y_init[i])
            gt_bb = labels_t[i]["boxes"]
            loss = torch.nn.L1Loss(size_average=False)(y_pred.float(), gt_bb.float())
            loss_list.append(loss)

        loss_dict: Dict[str, Union["torch.Tensor", int, List["torch.Tensor"]]] = dict()
        if reduction == "sum":
            loss_dict["torch.nn.L1Loss"] = sum(loss_list)
        elif reduction == "none":
            loss_dict["torch.nn.L1Loss"] = loss_list
        else:
            raise ValueError("Reduction not recognised.")

        return loss_dict, inputs_t, image_tensor_list_grad

    def loss_gradient(  # pylint: disable=W0613
        self, x: np.ndarray, y: Union[List[Dict[str, np.ndarray]], List[Dict[str, "torch.Tensor"]]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W.
                  - labels (Int64Tensor[N]): the predicted labels for each image.
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss gradients of the same shape as `x`.
        """
        grad_list = list()

        for i in range(x.shape[0]):

            x_i = x[[i]]
            y_i = [y[i]]

            output, _, image_tensor_list_grad = self._get_losses(x=x_i, y=y_i)

            # Compute the gradient and return
            loss = None
            for loss_name in self.attack_losses:
                if loss is None:
                    loss = output[loss_name]
                else:
                    loss = loss + output[loss_name]

            # Clean gradients
            self._model.zero_grad()

            # Compute gradients
            loss.backward(retain_graph=True)  # type: ignore

            for img in image_tensor_list_grad:
                if img.grad is not None:
                    gradients = img.grad.cpu().numpy().copy()
                else:
                    gradients = None
                grad_list.append(gradients)

        grads = np.array(grad_list)

        if grads.shape[0] == 1:
            grads_ = np.empty(len(grads), dtype=object)
            grads_[:] = list(grads)
            grads = grads_

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        if x.dtype != object:
            grads = np.array([i for i in grads], dtype=x.dtype)  # pylint: disable=R1721
            assert grads.shape == x.shape and grads.dtype == x.dtype

        return grads

    def _preprocess(self, img: "torch.Tensor") -> "torch.Tensor":
        """
        Preprocess image before forward pass, this is the same preprocessing used during training, please refer to
        collate function in train.py for reference

        :param img: Single frame od shape (nb_samples, height, width, nb_channels).
        :return: Preprocessed frame.
        """
        import torch  # lgtm [py/repeated-import]
        from torch.nn.functional import interpolate

        from art.preprocessing.standardisation_mean_std.pytorch import StandardisationMeanStdPyTorch

        if self.preprocessing is not None and isinstance(self.preprocessing, StandardisationMeanStdPyTorch):
            mean_np = self.preprocessing.mean
            std_np = self.preprocessing.std
        else:
            mean_np = np.ones((3, 1, 1))
            std_np = np.ones((3, 1, 1))
        mean = torch.from_numpy(mean_np).reshape((3, 1, 1))
        std = torch.from_numpy(std_np).reshape((3, 1, 1))
        img = img.permute(2, 0, 1)
        img = img * std + mean
        img = torch.unsqueeze(img, dim=0)
        img = interpolate(img, size=(self.input_shape[1], self.input_shape[2]), mode="bicubic")
        if self.clip_values is not None:
            img = torch.clamp(img, self.clip_values[0], self.clip_values[1])
        img = torch.squeeze(img)
        img = (img - mean) / std
        return img

    def _track_step(
        self, curr_frame: "torch.Tensor", prev_frame: "torch.Tensor", rect: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Track current frame.

        :param curr_frame: Current frame.
        :param prev_frame: Previous frame.
        :return: bounding box of previous frame
        """
        import torch  # lgtm [py/repeated-import]

        prev_bbox = rect

        k_context_factor = 2

        def compute_output_height_f(bbox_tight: "torch.Tensor") -> "torch.Tensor":
            """
            Compute height of search/target region.

            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].
            :return: Output height.
            """
            bbox_height = bbox_tight[3] - bbox_tight[1]
            output_height = k_context_factor * bbox_height

            return torch.maximum(torch.tensor(1.0).to(self.device), output_height)

        def compute_output_width_f(bbox_tight: "torch.Tensor") -> "torch.Tensor":
            """
            Compute width of search/target region.

            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].
            :return: Output width.
            """
            bbox_width = bbox_tight[2] - bbox_tight[0]
            output_width = k_context_factor * bbox_width

            return torch.maximum(torch.tensor(1.0).to(self.device), output_width)

        def get_center_x_f(bbox_tight: "torch.Tensor") -> "torch.Tensor":
            """
            Compute x-coordinate of the bounding box center.

            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].
            :return: x-coordinate of the bounding box center.
            """
            return (bbox_tight[0] + bbox_tight[2]) / 2.0

        def get_center_y_f(bbox_tight: "torch.Tensor") -> "torch.Tensor":
            """
            Compute y-coordinate of the bounding box center

            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].
            :return: y-coordinate of the bounding box center.
            """
            return (bbox_tight[1] + bbox_tight[3]) / 2.0

        def compute_crop_pad_image_location(
            bbox_tight: "torch.Tensor", image: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """
            Get the valid image coordinates for the context region in target or search region in full image

            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].
            :param image: Frame to be cropped and padded.
            :return: x-coordinate of the bounding box center.
            """

            # Center of the bounding box
            # bbox_center_x = bbox_tight.get_center_x()
            # bbox_center_y = bbox_tight.get_center_y()
            bbox_center_x = get_center_x_f(bbox_tight)
            bbox_center_y = get_center_y_f(bbox_tight)

            image_height = image.shape[0]
            image_width = image.shape[1]

            # Padded output width and height
            # output_width = bbox_tight.compute_output_width()
            # output_height = bbox_tight.compute_output_height()
            output_width = compute_output_width_f(bbox_tight)
            output_height = compute_output_height_f(bbox_tight)

            roi_left = torch.maximum(torch.tensor(0.0).to(self.device), bbox_center_x - (output_width / 2.0))
            roi_bottom = torch.maximum(torch.tensor(0.0).to(self.device), bbox_center_y - (output_height / 2.0))

            # New ROI width
            # -------------
            # 1. left_half should not go out of bound on the left side of the
            # image
            # 2. right_half should not go out of bound on the right side of the
            # image
            left_half = torch.minimum(output_width / 2.0, bbox_center_x)
            right_half = torch.minimum(output_width / 2.0, image_width - bbox_center_x)
            roi_width = torch.maximum(torch.tensor(1.0).to(self.device), left_half + right_half)

            # New ROI height
            # Similar logic applied that is applied for 'New ROI width'
            top_half = torch.minimum(output_height / 2.0, bbox_center_y)
            bottom_half = torch.minimum(output_height / 2.0, image_height - bbox_center_y)
            roi_height = torch.maximum(torch.tensor(1.0).to(self.device), top_half + bottom_half)

            # Padded image location in the original image
            # objPadImageLocation = BoundingBox(roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height)
            #
            # return objPadImageLocation
            return roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height

        def edge_spacing_x_f(bbox_tight: "torch.Tensor") -> "torch.Tensor":
            """
            Edge spacing X to take care of if search/target pad region goes out of bound.

            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].
            :return: Edge spacing X.
            """
            output_width = compute_output_width_f(bbox_tight)
            bbox_center_x = get_center_x_f(bbox_tight)

            return torch.maximum(torch.tensor(0.0).to(self.device), (output_width / 2) - bbox_center_x)

        def edge_spacing_y_f(bbox_tight: "torch.Tensor") -> "torch.Tensor":
            """
            Edge spacing X to take care of if search/target pad region goes out of bound.

            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].
            :return: Edge spacing X.
            """
            output_height = compute_output_height_f(bbox_tight)
            bbox_center_y = get_center_y_f(bbox_tight)

            return torch.maximum(torch.tensor(0.0).to(self.device), (output_height / 2) - bbox_center_y)

        def crop_pad_image(
            bbox_tight: "torch.Tensor", image: "torch.Tensor"
        ) -> Tuple[
            "torch.Tensor",
            Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"],
            "torch.Tensor",
            "torch.Tensor",
        ]:
            """
            Around the bounding box, we define a extra context factor of 2, which we will crop from the original image.

            :param bbox_tight: Coordinates of bounding box [x1, y1, x2, y2].
            :param image: Frame to be cropped and padded.
            :return: Cropped and Padded image.
            """
            import math
            import torch  # lgtm [py/repeated-import]

            pad_image_location = compute_crop_pad_image_location(bbox_tight, image)
            # roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
            # roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))
            roi_left = min(pad_image_location[0], (image.shape[1] - 1))  # type: ignore
            roi_bottom = min(pad_image_location[1], (image.shape[0] - 1))  # type: ignore
            # roi_width = min(image.shape[1], max(1.0, math.ceil(pad_image_location.x2 - pad_image_location.x1)))
            # roi_height = min(image.shape[0], max(1.0, math.ceil(pad_image_location.y2 - pad_image_location.y1)))
            roi_width = min(  # type: ignore
                image.shape[1], max(1, math.ceil(pad_image_location[2] - pad_image_location[0]))
            )
            roi_height = min(  # type: ignore
                image.shape[0], max(1, math.ceil(pad_image_location[3] - pad_image_location[1]))
            )

            roi_bottom_int = int(roi_bottom)
            roi_bottom_height_int = roi_bottom_int + roi_height

            roi_left_int = int(roi_left)
            roi_left_width_int = roi_left_int + roi_width

            cropped_image = image[roi_bottom_int:roi_bottom_height_int, roi_left_int:roi_left_width_int]

            # output_width = max(math.ceil(bbox_tight.compute_output_width()), roi_width)
            # output_height = max(math.ceil(bbox_tight.compute_output_height()), roi_height)
            output_width = max(math.ceil(compute_output_width_f(bbox_tight)), roi_width)
            output_height = max(math.ceil(compute_output_height_f(bbox_tight)), roi_height)
            if image.ndim > 2:
                # output_image = np.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
                output_image = torch.zeros((int(output_height), int(output_width), image.shape[2]), dtype=image.dtype)
            else:
                # output_image = np.zeros((int(output_height), int(output_width)), dtype=image.dtype)
                output_image = torch.zeros((int(output_height), int(output_width)), dtype=image.dtype)

            # edge_spacing_x = min(bbox_tight.edge_spacing_x(), (image.shape[1] - 1))
            # edge_spacing_y = min(bbox_tight.edge_spacing_y(), (image.shape[0] - 1))
            edge_spacing_x = min(edge_spacing_x_f(bbox_tight), (image.shape[1] - 1))  # type: ignore
            edge_spacing_y = min(edge_spacing_y_f(bbox_tight), (image.shape[0] - 1))  # type: ignore

            # rounding should be done to match the width and height
            output_image[
                int(edge_spacing_y) : int(edge_spacing_y) + cropped_image.shape[0],
                int(edge_spacing_x) : int(edge_spacing_x) + cropped_image.shape[1],
            ] = cropped_image

            return output_image, pad_image_location, edge_spacing_x, edge_spacing_y

        target_pad, _, _, _ = crop_pad_image(prev_bbox, prev_frame)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = crop_pad_image(prev_bbox, curr_frame)

        target_pad_in = self._preprocess(target_pad).unsqueeze(0).to(self.device)
        cur_search_region_in = self._preprocess(cur_search_region).unsqueeze(0).to(self.device)

        pred_bb = self._model.forward(target_pad_in.float(), cur_search_region_in.float())

        pred_bb = torch.squeeze(pred_bb)

        k_scale_factor = 10
        height = cur_search_region.shape[0]
        width = cur_search_region.shape[1]

        # Normalize the image bounding box
        pred_bb[0] = pred_bb[0] / k_scale_factor * width
        pred_bb[2] = pred_bb[2] / k_scale_factor * width
        pred_bb[1] = pred_bb[1] / k_scale_factor * height
        pred_bb[3] = pred_bb[3] / k_scale_factor * height

        # brings gradients to zero
        # pred_bb = torch.round(pred_bb)

        # move the bounding box to target/search region coordinates
        raw_image = curr_frame
        pred_bb[0] = max(0.0, pred_bb[0] + search_location[0] - edge_spacing_x)
        pred_bb[1] = max(0.0, pred_bb[1] + search_location[1] - edge_spacing_y)
        pred_bb[2] = min(raw_image.shape[1], pred_bb[2] + search_location[0] - edge_spacing_x)
        pred_bb[3] = min(raw_image.shape[0], pred_bb[3] + search_location[1] - edge_spacing_y)

        # brings gradients to zero
        # pred_bb = torch.round(pred_bb)

        return pred_bb

    def _track(self, x: "torch.Tensor", y_init: "torch.Tensor") -> "torch.Tensor":
        """
        Track object across frames.

        :param x: A single video of shape (nb_frames, nb_height, nb_width, nb_channels)
        :param y_init: Initial bounding box around object on the first frame of `x`.
        :return: Predicted bounding box coordinates for all frames of shape (nb_frames, 4) in format [x1, y1, x2, y2].
        """
        import torch  # lgtm [py/repeated-import]

        num_frames = x.shape[0]
        prev = x[0]
        bbox_0 = y_init
        y_pred_list = [y_init]

        for i in range(1, num_frames):
            curr = x[i]
            bbox_0 = self._track_step(curr, prev, bbox_0)
            bbox = bbox_0
            prev = curr

            y_pred_list.append(bbox)

        y_pred = torch.stack(y_pred_list)

        return y_pred

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, nb_frames, height, width, nb_channels).
        :param batch_size: Batch size.

        :Keyword Arguments:
            * *y_init* (``np.ndarray``) --
              Initial box around object to be tracked as [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
              0 <= y1 < y2 <= H.

        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one dictionary for each input image. The keys of
                 the dictionary are:

                  - boxes [N_FRAMES, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                         0 <= y1 < y2 <= H.
                  - labels [N_FRAMES]: the labels for each image, default 0.
                  - scores [N_FRAMES]: the scores or each prediction, default 1.
        """
        import torch  # lgtm [py/repeated-import]

        self._model.eval()
        if hasattr(self._model, "freeze"):
            self._model.freeze()

        y_init = kwargs.get("y_init")
        if y_init is None:  # pragma: no cover
            raise ValueError("y_init is a required argument for method `predict`.")

        if isinstance(y_init, np.ndarray):
            y_init = torch.from_numpy(y_init).to(self.device).float()
        else:
            y_init = y_init.to(self.device).float()

        predictions = list()

        for i in range(x.shape[0]):
            if isinstance(x, np.ndarray):
                x_i = torch.from_numpy(x[i]).to(self.device)
            else:
                x_i = x[i].to(self.device)

            # Apply preprocessing
            x_i, _ = self._apply_preprocessing(x_i, y=None, fit=False, no_grad=False)

            y_pred = self._track(x=x_i, y_init=y_init[i])

            prediction_dict = dict()
            if isinstance(x, np.ndarray):
                prediction_dict["boxes"] = y_pred.detach().cpu().numpy()
            else:
                prediction_dict["boxes"] = y_pred
            predictions.append(prediction_dict)

        return predictions

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        """
        Not implemented.
        """
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        """
        Not implemented.
        """
        raise NotImplementedError

    def compute_losses(self, x: np.ndarray, y: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        """
        Not implemented.
        """
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: List[Dict[str, np.ndarray]], **kwargs) -> np.ndarray:
        """
        Not implemented.
        """
        raise NotImplementedError

    def init(self, image: "PIL.JpegImagePlugin.JpegImageFile", box: np.ndarray):
        """
        Method `init` for GOT-10k trackers.

        :param image: Current image.
        :return: Predicted box.
        """
        import torch  # lgtm [py/repeated-import]

        self.prev = np.array(image) / 255.0
        if self.clip_values is not None:
            self.prev = self.prev * self.clip_values[1]
        self.box = torch.from_numpy(np.array([box[0], box[1], box[2] + box[0], box[3] + box[1]])).to(self.device)

    def update(self, image: np.ndarray) -> np.ndarray:
        """
        Method `update` for GOT-10k trackers.

        :param image: Current image.
        :return: Predicted box.
        """
        import torch  # lgtm [py/repeated-import]

        curr = torch.from_numpy(np.array(image) / 255.0)
        if self.clip_values is not None:
            curr = curr * self.clip_values[1]
        curr = curr.to(self.device)
        prev = torch.from_numpy(self.prev).to(self.device)

        curr, _ = self._apply_preprocessing(curr, y=None, fit=False)

        self.box = self._track_step(curr, prev, self.box)
        self.prev = curr.cpu().detach().numpy()

        box_return = self.box.cpu().detach().numpy()
        box_return = np.array(
            [box_return[0], box_return[1], box_return[2] - box_return[0], box_return[3] - box_return[1]]
        )

        return box_return

    def track(self, img_files: List[str], box: np.ndarray, visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method `track` for GOT-10k toolkit trackers (MIT licence).

        :param img_files: Image files.
        :param box: Initial boxes.
        :param visualize: Visualise tracking.
        """
        from got10k.utils.viz import show_frame
        from PIL import Image

        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for i_f, img_file in enumerate(img_files):
            image = Image.open(img_file)
            if not image.mode == "RGB":
                image = image.convert("RGB")

            start_time = time.time()
            if i_f == 0:
                self.init(image, box)
            else:
                boxes[i_f, :] = self.update(image)
            times[i_f] = time.time() - start_time

            if visualize:
                show_frame(image, boxes[i_f, :])

        return boxes, times
