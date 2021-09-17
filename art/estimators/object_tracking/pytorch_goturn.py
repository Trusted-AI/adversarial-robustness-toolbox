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
"""
This module implements the task specific estimator for PyTorch Goturn object detectors.
"""
import os
import sys
import logging
from typing import List, Dict, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.pytorch import PyTorchEstimator
from art import config

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.defences.postprocessor.postprocessor import Postprocessor

logger = logging.getLogger(__name__)


class PyTorchGoturn(ObjectDetectorMixin, PyTorchEstimator):
    """
    This module implements the task specific estimator for PyTorch object detectors.
    """

    estimator_params = PyTorchEstimator.estimator_params + ["attack_losses"]

    def __init__(
        self,
        goturn_path: Optional[str] = None,
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = None,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param goturn_path: Path to GOTURN repository.
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
        from git import Repo
        import torch  # lgtm [py/repeated-import]
        import torchvision  # lgtm [py/repeated-import]

        torch_version = list(map(int, torch.__version__.lower().split("+")[0].split(".")))
        torchvision_version = list(map(int, torchvision.__version__.lower().split("+")[0].split(".")))
        assert torch_version[0] == 1 and torch_version[1] == 4, "PyTorchGoturn requires torch==1.4"
        assert torchvision_version[0] == 0 and torchvision_version[1] == 5, "PyTorchGoturn requires torchvision==0.5"

        self.goturn_path = goturn_path

        # Set device
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device("cuda:{}".format(cuda_idx))

        if goturn_path is None:
            self.goturn_path = os.path.join(config.ART_DATA_PATH, "goturn")

            if not os.path.isdir(self.goturn_path):
                git_url = "git@github.com:nrupatunga/goturn-pytorch.git"
                Repo.clone_from(git_url, self.goturn_path)

        sys.path.insert(0, config.ART_DATA_PATH + "/goturn/src")
        sys.path.insert(0, config.ART_DATA_PATH + "/goturn/src/scripts")

        from scripts.train import GoturnTrain
        from pathlib import Path

        model_dir = Path(os.path.join(self.goturn_path, "src", "goturn", "models"))
        ckpt_dir = model_dir.joinpath("checkpoints")
        ckpt_path = next(ckpt_dir.glob("*.ckpt"))

        ckpt_mod = torch.load(
            "/home/bbuesser/.art/data/goturn/src/goturn/models/checkpoints/_ckpt_epoch_3.ckpt", map_location="cpu"
        )
        ckpt_mod["hparams"]["pretrained_model"] = os.path.join(
            self.goturn_path, "src", "goturn", "models", "pretrained", "caffenet_weights.npy"
        )
        torch.save(ckpt_mod, "/home/bbuesser/.art/data/goturn/src/goturn/models/checkpoints/_ckpt_epoch_3.ckpt")

        model = GoturnTrain.load_from_checkpoint(ckpt_path)
        model.to(self._device)
        model.eval()
        model.freeze()

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._input_shape = None

        if self.clip_values is not None:
            if self.clip_values[0] != 0:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, 255).")
            if self.clip_values[1] != 255:
                raise ValueError("This classifier requires un-normalized input images with clip_vales=(0, 255).")

        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        self.attack_losses: Tuple[str, ...] = ["torch.nn.L1Loss"]

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
        self, x: np.ndarray, y: Union[List[Dict[str, np.ndarray]], List[Dict[str, "torch.Tensor"]]]
    ) -> Tuple[Dict[str, "torch.Tensor"], List["torch.Tensor"], List["torch.Tensor"]]:
        """
        Get the loss tensor output of the model including all preprocessing.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The fields of the Dict are as
                  follows:

                  - boxes (FloatTensor[N, 4]): the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                               0 <= y1 < y2 <= H.
                  - labels (Int64Tensor[N]): the labels for each image
        :return: Loss gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]

        self._model.train()

        # Apply preprocessing
        if self.all_framework_preprocessing:
            print("self.all_framework_preprocessing")
            if isinstance(x, torch.Tensor):
                raise NotImplementedError

            if y is not None and isinstance(y[0]["boxes"], np.ndarray):
                print("isinstance(y[0][\"boxes\"], np.ndarray)")
                y_tensor = list()
                for i, y_i in enumerate(y):
                    y_t = dict()
                    y_t["boxes"] = torch.from_numpy(y_i["boxes"]).type(torch.float).to(self._device)
                    y_t["labels"] = torch.from_numpy(y_i["labels"]).type(torch.int64).to(self._device)
                    if "masks" in y_i:
                        y_t["masks"] = torch.from_numpy(y_i["masks"]).type(torch.int64).to(self._device)
                    y_tensor.append(y_t)
            else:
                y_tensor = y

            # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            image_tensor_list_grad = list()
            y_preprocessed = list()
            inputs_t = list()

            for i in range(x.shape[0]):
                if self.clip_values is not None:
                    # x_grad = transform(x[i] / self.clip_values[1]).to(self._device)
                    x_grad = torch.from_numpy(x[i]).to(self._device).float()
                else:
                    # x_grad = transform(x[i]).to(self._device)
                    x_grad = torch.from_numpy(x[i]).to(self._device).float()
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

        if isinstance(y_preprocessed, np.ndarray):
            labels_t = torch.from_numpy(y_preprocessed).to(self._device)  # type: ignore
        else:
            labels_t = y_preprocessed  # type: ignore

        # self._model.eval()
        # self._model.freeze()

        # y_init = torch.from_numpy(np.array([[72, 89, 121, 146], [160, 100, 180, 146]])).float()
        y_init = torch.from_numpy(np.array([[42, 89, 121, 146], [160, 100, 180, 146]])).float()

        loss_list = list()

        for i in range(x.shape[0]):
            print('i', i)
            x_i = inputs_t[i]
            # x_i = torch.from_numpy(x[i]).float()
            # x_i.requires_grad = True
            # self.x_i_grad = x_i

            y_pred = self.track(x=x_i, y_init=y_init[i])

            # import cv2
            # for i in range(0, 40):
            #     curr = x_i[i]
            #     bbox_0 = y_pred[i]
            #     bbox = bbox_0
            #     prev = curr
            #
            #     curr_dbg = np.copy(curr.detach().numpy())
            #     # curr = x_i.detach().numpy()
            #     mean_np = np.array([104, 117, 123])
            #     curr_dbg = curr_dbg + mean_np
            #     curr_dbg = curr_dbg.astype(np.uint8)
            #
            #     print(bbox)
            #
            #     # curr_dbg = np.copy(curr)
            #     curr_dbg = cv2.rectangle(
            #         curr_dbg, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2
            #     )
            #
            #     cv2.imshow("image", curr_dbg)
            #     # cv2.waitKey(20)
            #     cv2.waitKey()

            gt_bb = labels_t[i]['boxes']
            loss = torch.nn.L1Loss(size_average=False)(y_pred.float(), gt_bb.float())
            loss_list.append(loss)
            # print('loss', loss)
            # loss.backward()

            # gt_bb = labels_t[i]['boxes']
            # loss = torch.nn.L1Loss(size_average=False)(y_pred.float(), gt_bb.float())
            # print('loss', loss)
            # loss.backward()

            # print('self.target_pad_in.grad', self.target_pad_in.grad)
            # print('self.target_pad.grad', self.target_pad.grad)
            # print('self.cur_search_region.grad', self.cur_search_region.grad)
            # print('self.x_grad.grad', self.x_grad.grad)
            # print('image_tensor_list_grad[i].requires_grad', image_tensor_list_grad[i].requires_grad)
            # print('image_tensor_list_grad[i].grad', image_tensor_list_grad[i].grad)
            # print('image_tensor_list_grad[i].grad min', np.min(image_tensor_list_grad[i].grad.detach().numpy()))
            # print('image_tensor_list_grad[i].grad max', np.max(image_tensor_list_grad[i].grad.detach().numpy()))
            # asdf

        loss = {"torch.nn.L1Loss": sum(loss_list)}

        # self._model.zero_grad()
        # loss["torch.nn.L1Loss"].backward()
        # print('image_tensor_list_grad[0].grad min', np.min(image_tensor_list_grad[0].grad.detach().numpy()))
        # print('image_tensor_list_grad[0].grad max', np.max(image_tensor_list_grad[0].grad.detach().numpy()))

        return loss, inputs_t, image_tensor_list_grad

    def loss_gradient(  # pylint: disable=W0613
        self, x: np.ndarray, y: Union[List[Dict[str, np.ndarray]], List[Dict[str, "torch.Tensor"]]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss gradients of the same shape as `x`.
        """
        import torch  # lgtm [py/repeated-import]

        grad_list = list()

        for i in range(x.shape[0]):

            x_i = x[[i]]
            y_i = [y[i]]

            output, inputs_t, image_tensor_list_grad = self._get_losses(x=x_i, y=y_i)

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

            # loss.backward()
            # print('image_tensor_list_grad[0].grad min', np.min(image_tensor_list_grad[0].grad.detach().numpy()))
            # print('image_tensor_list_grad[0].grad max', np.max(image_tensor_list_grad[0].grad.detach().numpy()))
            # sdsdgf

            if isinstance(x, np.ndarray):
                for img in image_tensor_list_grad:
                    gradients = img.grad.cpu().numpy().copy()
                    # gradients = img.grad.cpu().detach().numpy()
                    # print(gradients)
                    # print('gradients.grad min', np.min(gradients))
                    # print('gradients.grad max', np.max(gradients))
                    # ghj
                    grad_list.append(gradients)
            else:
                for img in inputs_t:
                    gradients = img.grad.copy()
                    grad_list.append(gradients)

        if isinstance(x, np.ndarray):
            # grads = np.stack(grad_list, axis=0)
            grads = np.array(grad_list, dtype=object)
            # grads = np.transpose(grads, (0, 2, 3, 1))

        # if self.clip_values is not None:
        #     grads = grads / self.clip_values[1]

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads

    def preprocess(self, im):
        """
        preprocess image before forward pass, this is the same
        preprocessing used during training, please refer to collate function
        in train.py for reference
        @image: input image
        """
        import torch
        from torch.nn.functional import interpolate

        mean_np = np.array([104, 117, 123])
        mean = torch.from_numpy(mean_np).reshape((3, 1, 1))
        im = im.permute(2, 0, 1)
        im = im + mean
        im = torch.unsqueeze(im, dim=0)
        im = interpolate(im, size=(227, 227))
        im = torch.squeeze(im)
        im = im - mean
        return im

    def _track(self, curr_frame, prev_frame, rect):
        """track current frame
        @curr_frame: current frame
        @prev_frame: prev frame
        @rect: bounding box of previous frame
        """
        import torch
        from goturn.helper.BoundingBox import BoundingBox

        prev_bbox = rect

        kContextFactor = 2

        def compute_output_height_f(bbox_tight):
            """height of search/target region"""
            bbox_height = bbox_tight[3] - bbox_tight[1]
            output_height = kContextFactor * bbox_height

            return max(1.0, output_height)

        def compute_output_width_f(bbox_tight):
            """width of search/target region"""
            bbox_width = bbox_tight[2] - bbox_tight[0]
            output_width = kContextFactor * bbox_width

            return max(1.0, output_width)

        def get_center_x_f(bbox_tight):
            """x-coordinate of the bounding box center """
            return (bbox_tight[0] + bbox_tight[2]) / 2.0

        def get_center_y_f(bbox_tight):
            """y-coordinate of the bounding box center """
            return (bbox_tight[1] + bbox_tight[3]) / 2.0

        def computeCropPadImageLocation(bbox_tight, image):
            """Get the valid image coordinates for the context region in target
            or search region in full image
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

            roi_left = max(0.0, bbox_center_x - (output_width / 2.))
            roi_bottom = max(0.0, bbox_center_y - (output_height / 2.))

            # New ROI width
            # -------------
            # 1. left_half should not go out of bound on the left side of the
            # image
            # 2. right_half should not go out of bound on the right side of the
            # image
            left_half = min(output_width / 2., bbox_center_x)
            right_half = min(output_width / 2., image_width - bbox_center_x)
            roi_width = max(1.0, left_half + right_half)

            # New ROI height
            # Similar logic applied that is applied for 'New ROI width'
            top_half = min(output_height / 2., bbox_center_y)
            bottom_half = min(output_height / 2., image_height - bbox_center_y)
            roi_height = max(1.0, top_half + bottom_half)

            # Padded image location in the original image
            objPadImageLocation = BoundingBox(roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height)

            return objPadImageLocation

        def edge_spacing_x_f(bbox_tight):
            """Edge spacing X to take care of if search/target pad region goes
            out of bound
            """
            output_width = compute_output_width_f(bbox_tight)
            bbox_center_x = get_center_x_f(bbox_tight)

            return max(0.0, (output_width / 2) - bbox_center_x)

        def edge_spacing_y_f(bbox_tight):
            """Edge spacing X to take care of if search/target pad region goes
            out of bound
            """
            output_height = compute_output_height_f(bbox_tight)
            bbox_center_y = get_center_y_f(bbox_tight)

            return max(0.0, (output_height / 2) - bbox_center_y)

        def cropPadImage(bbox_tight, image, dbg=False, viz=None):
            """ Around the bounding box, we define a extra context factor of 2,
            which we will crop from the original image
            """
            import math
            import torch

            pad_image_location = computeCropPadImageLocation(bbox_tight, image)
            roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
            roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))
            roi_width = min(image.shape[1], max(1.0, math.ceil(pad_image_location.x2 - pad_image_location.x1)))
            roi_height = min(image.shape[0], max(1.0, math.ceil(pad_image_location.y2 - pad_image_location.y1)))

            err = 0.000000001  # To take care of floating point arithmetic errors
            cropped_image = image[int(roi_bottom + err):int(roi_bottom + roi_height),
                            int(roi_left + err):int(roi_left + roi_width)]
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
            edge_spacing_x = min(edge_spacing_x_f(bbox_tight), (image.shape[1] - 1))
            edge_spacing_y = min(edge_spacing_y_f(bbox_tight), (image.shape[0] - 1))

            # rounding should be done to match the width and height
            output_image[int(edge_spacing_y):int(edge_spacing_y) + cropped_image.shape[0],
            int(edge_spacing_x):int(edge_spacing_x) + cropped_image.shape[1]] = cropped_image

            return output_image, pad_image_location, edge_spacing_x, edge_spacing_y

        target_pad, _, _, _ = cropPadImage(prev_bbox, prev_frame)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(prev_bbox, curr_frame)

        # target_pad.requires_grad = True
        # self.target_pad = target_pad
        # cur_search_region.requires_grad = True
        # self.cur_search_region = cur_search_region

        target_pad_in = self.preprocess(target_pad).unsqueeze(0)
        cur_search_region_in = self.preprocess(cur_search_region).unsqueeze(0)

        # target_pad_in.requires_grad = True
        # self.target_pad_in = target_pad_in

        pred_bb = self._model.forward(target_pad_in, cur_search_region_in)

        pred_bb = torch.squeeze(pred_bb)

        kScaleFactor = 10
        height = cur_search_region.shape[0]
        width = cur_search_region.shape[1]

        """Normalize the image bounding box"""
        pred_bb[0] = pred_bb[0] / kScaleFactor * width
        pred_bb[2] = pred_bb[2] / kScaleFactor * width
        pred_bb[1] = pred_bb[1] / kScaleFactor * height
        pred_bb[3] = pred_bb[3] / kScaleFactor * height

        # brings gradients to zero
        # pred_bb = torch.round(pred_bb)

        """move the bounding box to target/search region coordinates"""
        raw_image = curr_frame
        pred_bb[0] = max(0.0, pred_bb[0] + search_location.x1 - edge_spacing_x)
        pred_bb[1] = max(0.0, pred_bb[1] + search_location.y1 - edge_spacing_y)
        pred_bb[2] = min(raw_image.shape[1], pred_bb[2] + search_location.x1 - edge_spacing_x)
        pred_bb[3] = min(raw_image.shape[0], pred_bb[3] + search_location.y1 - edge_spacing_y)

        # brings gradients to zero
        # pred_bb = torch.round(pred_bb)

        return pred_bb

    def track(self, x, y_init):
        """Track"""
        import torch

        # x.requires_grad = True
        # self.x_grad = x

        num_frames = x.shape[0]
        prev = x[0]
        bbox_0 = y_init
        y_pred_list = [y_init]

        for i in range(1, num_frames):
            curr = x[i]
            bbox_0 = self._track(curr, prev, bbox_0)
            bbox = bbox_0
            prev = curr

            y_pred_list.append(bbox)

        y_pred = torch.stack(y_pred_list)

        return y_pred

    def predict(self, x: np.ndarray, batch_size: int = 128, **kwargs) -> List[Dict[str, np.ndarray]]:
        """
        Perform prediction for a batch of inputs.

        :param x: Samples of shape (nb_samples, height, width, nb_channels).
        :param batch_size: Batch size.
        :return: Predictions of format `List[Dict[str, np.ndarray]]`, one for each input image. The fields of the Dict
                 are as follows:

                 - boxes [N, 4]: the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H.
                 - labels [N]: the labels for each image
                 - scores [N]: the scores or each prediction.
        """
        import torch
        self._model.eval()
        self._model.freeze()

        y_init = torch.from_numpy(np.array([[72, 89, 121, 146], [160, 100, 180, 146]])).float()

        predictions = list()

        for i in range(x.shape[0]):
            x_i = torch.from_numpy(x[i])

            # Apply preprocessing
            x_i, _ = self._apply_preprocessing(x_i, y=None, fit=False)

            y_pred = self.track(x=x_i, y_init=y_init[i])

            prediction_dict = dict()
            prediction_dict["boxes"] = y_pred
            prediction_dict["labels"] = np.zeros((y_pred.shape[0],))
            prediction_dict["scores"] = np.ones_like((y_pred.shape[0],))
            predictions.append(prediction_dict)

        return predictions

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        raise NotImplementedError

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def compute_losses(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute all loss components.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices
                  of shape `(nb_samples,)`.
        :return: Dictionary of loss components.
        """
        output, _, _ = self._get_losses(x=x, y=y)
        return output

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the loss of the neural network for samples `x`.
        """
        import torch  # lgtm [py/repeated-import]

        output, _, _ = self._get_losses(x=x, y=y)

        # Compute the gradient and return
        loss = None
        for loss_name in self.attack_losses:
            if loss is None:
                loss = output[loss_name]
            else:
                loss = loss + output[loss_name]

        assert loss is not None

        if isinstance(x, torch.Tensor):
            return loss

        return loss.detach().cpu().numpy()
