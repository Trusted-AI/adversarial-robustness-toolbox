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
This module implements the task specific estimator for DEtection TRansformer (DETR) in PyTorch.

 | Paper link: https://arxiv.org/abs/2005.12872
"""
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np

from art.estimators.object_detection.object_detector import ObjectDetectorMixin
from art.estimators.pytorch import PyTorchEstimator

if TYPE_CHECKING:
    # pylint: disable=C0412
    import torch

    from art.defences.postprocessor.postprocessor import Postprocessor
    from art.defences.preprocessor.preprocessor import Preprocessor
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE

logger = logging.getLogger(__name__)


def box_cxcywh_to_xyxy(x: "torch.Tensor"):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/box_ops.py)
    """
    import torch

    x_c, y_c, width, height = x.unbind(1)
    box = [(x_c - 0.5 * width), (y_c - 0.5 * height), (x_c + 0.5 * width), (y_c + 0.5 * height)]
    return torch.stack(box, dim=1)


def box_xyxy_to_cxcywh(x: "torch.Tensor"):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/box_ops.py)
    """
    import torch

    x_0, y_0, x_1, y_1 = x.unbind(-1)
    box = [(x_0 + x_1) / 2, (y_0 + y_1) / 2, (x_1 - x_0), (y_1 - y_0)]
    return torch.stack(box, dim=-1)


def rescale_bboxes(out_bbox: "torch.Tensor", size: Tuple[int, int]):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (inference notebook)
    """
    import torch

    img_w, img_h = size
    box = box_cxcywh_to_xyxy(out_bbox)
    box = box * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return box


def revert_rescale_bboxes(out_bbox: "torch.Tensor", size: Tuple[int, int]):
    """
    Adapted rom DETR source: https://github.com/facebookresearch/detr
    (inference notebook)
    This method reverts bounding box rescaling to match input image size
    """
    import torch

    img_w, img_h = size
    box = out_bbox / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    box = box_xyxy_to_cxcywh(box)
    return box


def box_iou(boxes1: "torch.Tensor", boxes2: "torch.Tensor"):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/box_ops.py)
    """
    import torch
    from torchvision.ops.boxes import box_area

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    l_t = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    r_b = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    w_h = (r_b - l_t).clamp(min=0)  # [N,M,2]
    inter = w_h[:, :, 0] * w_h[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1: "torch.Tensor", boxes2: "torch.Tensor"):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/box_ops.py)
    """
    import torch

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    l_t = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    r_b = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    w_h = (r_b - l_t).clamp(min=0)  # [N,M,2]
    area = w_h[:, :, 0] * w_h[:, :, 1]

    return iou - (area - union) / area


class PyTorchDetectionTransformer(ObjectDetectorMixin, PyTorchEstimator):
    """
    This class implements a model-specific object detector using DEtection TRansformer (DETR)
    and PyTorch following the input and output formats of torchvision.
    """

    MIN_IMAGE_SIZE = 800
    MAX_IMAGE_SIZE = 1333
    estimator_params = PyTorchEstimator.estimator_params + ["attack_losses"]

    def __init__(
        self,
        model: "torch.nn.Module" = None,
        input_shape: Tuple[int, ...] = (3, 800, 800),
        clip_values: Optional["CLIP_VALUES_TYPE"] = None,
        channels_first: Optional[bool] = True,
        preprocessing_defences: Union["Preprocessor", List["Preprocessor"], None] = None,
        postprocessing_defences: Union["Postprocessor", List["Postprocessor"], None] = None,
        preprocessing: "PREPROCESSING_TYPE" = None,
        attack_losses: Tuple[str, ...] = (
            "loss_ce",
            "loss_bbox",
            "loss_giou",
        ),
        device_type: str = "gpu",
    ):
        """
        Initialization.

        :param model: DETR model. The output of the model is `List[Dict[Tensor]]`, one for each input image. The
                      fields of the Dict are as follows:

                      - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                        between 0 and H and 0 and W
                      - labels (Tensor[N]): the predicted labels for each image
                      - scores (Tensor[N]): the scores or each prediction
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :param channels_first: Set channels first or last.
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the estimator.
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the estimator.
        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :param device_type: Type of device to be used for model and tensors, if `cpu` run on CPU, if `gpu` run on GPU
                            if available otherwise run on CPU.
        """
        import torch

        class HungarianMatcher(torch.nn.Module):
            """
            From DETR source: https://github.com/facebookresearch/detr
            (detr/models/matcher.py)
            """

            def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
                """Creates the matcher
                Params:
                    cost_class: This is the relative weight of the classification error
                                in the matching cost
                    cost_bbox:  This is the relative weight of the L1 error
                                of the bounding box coordinates in the matching cost
                    cost_giou:  This is the relative weight of the giou loss of the
                                bounding box in the matching cost
                """
                super().__init__()
                self.cost_class = cost_class
                self.cost_bbox = cost_bbox
                self.cost_giou = cost_giou
                assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

            def forward(self, outputs, targets):
                """Performs the matching
                Params:
                    outputs: This is a dict that contains at least these entries:
                        "pred_logits":  Tensor of dim [batch_size, num_queries, num_classes]
                                        with the classification logits
                        "pred_boxes":   Tensor of dim [batch_size, num_queries, 4]
                                        with the predicted box coordinates
                    targets: This is a list of targets (len(targets) = batch_size),
                        where each target is a dict containing:
                        "labels":   Tensor of dim [num_target_boxes] (where num_target_boxes
                                    is the number of ground-truth
                                    objects in the target) containing the class labels
                        "boxes":    Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                Returns:
                    A list of size batch_size, containing tuples of (index_i, index_j) where:
                        - index_i is the indices of the selected predictions (in order)
                        - index_j is the indices of the corresponding selected targets (in order)
                    For each batch element, it holds:
                        len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
                """
                import torch
                from scipy.optimize import linear_sum_assignment

                batch_size, num_queries = outputs["pred_logits"].shape[:2]

                # We flatten to compute the cost matrices in a batch
                out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
                out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

                # Also concat the target labels and boxes
                tgt_ids = torch.cat([v["labels"] for v in targets])
                tgt_bbox = torch.cat([v["boxes"] for v in targets])

                # Compute the classification cost. Contrary to the loss, we don't use the NLL,
                # but approximate it in 1 - proba[target class].
                # The 1 is a constant that doesn't change the matching, it can be ommitted.
                cost_class = -out_prob[:, tgt_ids]

                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

                # Compute the giou cost betwen boxes
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

                # Final cost matrix
                cost_matrix = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

                sizes = [len(v["boxes"]) for v in targets]
                indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
                return [
                    (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
                ]

        class SetCriterion(torch.nn.Module):
            """
            From DETR source: https://github.com/facebookresearch/detr
            (detr/models/detr.py)
            """

            def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
                """Create the criterion.
                Parameters:
                    num_classes: number of object categories, omitting the special no-object category
                    matcher: module able to compute a matching between targets and proposals
                    weight_dict: dict containing as key the names of the losses and as values their relative weight.
                    eos_coef: relative classification weight applied to the no-object category
                    losses: list of all the losses to be applied. See get_loss for list of available losses.
                """
                import torch

                super().__init__()
                self.num_classes = num_classes
                self.matcher = matcher
                self.weight_dict = weight_dict
                self.eos_coef = eos_coef
                self.losses = losses
                empty_weight = torch.ones(self.num_classes + 1)
                empty_weight[-1] = self.eos_coef
                self.register_buffer("empty_weight", empty_weight)

            @staticmethod
            def dice_loss(inputs, targets, num_boxes):
                """
                From DETR source: https://github.com/facebookresearch/detr
                (detr/models/segmentation.py)
                """
                inputs = inputs.sigmoid()
                inputs = inputs.flatten(1)
                numerator = 2 * (inputs * targets).sum(1)
                denominator = inputs.sum(-1) + targets.sum(-1)
                loss = 1 - (numerator + 1) / (denominator + 1)
                return loss.sum() / num_boxes

            @staticmethod
            def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
                """
                From DETR source: https://github.com/facebookresearch/detr
                (detr/models/segmentation.py)
                """
                import torch

                prob = inputs.sigmoid()
                ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
                p_t = prob * targets + (1 - prob) * (1 - targets)
                loss = ce_loss * ((1 - p_t) ** gamma)

                if alpha >= 0:
                    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                    loss = alpha_t * loss

                return loss.mean(1).sum() / num_boxes

            def loss_labels(self, outputs, targets, indices):
                """Classification loss (NLL)
                targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
                """
                import torch

                src_logits = outputs["pred_logits"]

                idx = self._get_src_permutation_idx(indices)
                target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
                target_classes = torch.full(
                    src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
                )
                target_classes[idx] = target_classes_o

                loss_ce = torch.nn.functional.cross_entropy(
                    src_logits.transpose(1, 2), target_classes, self.empty_weight
                )
                losses = {"loss_ce": loss_ce}
                return losses

            @staticmethod
            def loss_cardinality(outputs, targets):
                """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
                This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
                """
                import torch

                pred_logits = outputs["pred_logits"]
                device = pred_logits.device
                tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
                # Count the number of predictions that are NOT "no-object" (which is the last class)
                card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
                card_err = torch.nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
                losses = {"cardinality_error": card_err}
                return losses

            def loss_boxes(self, outputs, targets, indices, num_boxes):
                """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
                targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
                The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
                """
                import torch

                idx = self._get_src_permutation_idx(indices)
                src_boxes = outputs["pred_boxes"][idx]
                target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

                loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes, reduction="none")

                losses = {}
                losses["loss_bbox"] = loss_bbox.sum() / num_boxes

                loss_giou = 1 - torch.diag(
                    generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
                )
                losses["loss_giou"] = loss_giou.sum() / num_boxes
                return losses

            def loss_masks(self, outputs, targets, indices, num_boxes):
                """Compute the losses related to the masks: the focal loss and the dice loss.
                targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
                """
                import torchvision

                src_idx = self._get_src_permutation_idx(indices)
                tgt_idx = self._get_tgt_permutation_idx(indices)
                src_masks = outputs["pred_masks"]
                src_masks = src_masks[src_idx]
                masks = [t["masks"] for t in targets]
                # TODO use valid to mask invalid areas due to padding in loss
                target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
                target_masks = target_masks.to_device(src_masks)
                target_masks = target_masks[tgt_idx]

                # upsample predictions to the target size
                src_masks = torchvision.ops.misc.interpolate(
                    src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
                )
                src_masks = src_masks[:, 0].flatten(1)

                target_masks = target_masks.flatten(1)
                target_masks = target_masks.view(src_masks.shape)
                losses = {
                    "loss_mask": self.sigmoid_focal_loss(src_masks, target_masks, num_boxes),
                    "loss_dice": self.dice_loss(src_masks, target_masks, num_boxes),
                }
                return losses

            @staticmethod
            def _get_src_permutation_idx(indices):
                """
                permute predictions following indices
                """
                import torch

                batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
                src_idx = torch.cat([src for (src, _) in indices])
                return batch_idx, src_idx

            @staticmethod
            def _get_tgt_permutation_idx(indices):
                """
                permute targets following indices
                """
                import torch

                batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
                tgt_idx = torch.cat([tgt for (_, tgt) in indices])
                return batch_idx, tgt_idx

            def get_loss(self, loss, outputs, targets, indices, num_boxes):
                """
                Get the Hungarian Loss
                """
                if loss == "labels":
                    return self.loss_labels(outputs, targets, indices)
                if loss == "cardinality":
                    with torch.no_grad():
                        return self.loss_cardinality(outputs, targets)
                if loss == "boxes":
                    return self.loss_boxes(outputs, targets, indices, num_boxes)
                if loss == "masks":
                    return self.loss_masks(outputs, targets, indices, num_boxes)
                raise ValueError("No loss selected.")

            def forward(self, outputs, targets):
                """This performs the loss computation.
                Parameters:
                    outputs: dict of tensors, see the output specification of the model for the format
                    targets: list of dicts, such that len(targets) == batch_size.
                            The expected keys in each dict depends on the losses applied, see each loss' doc
                """
                import torch

                outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

                # Retrieve the matching between the outputs of the last layer and the targets
                with torch.no_grad():
                    indices = self.matcher(outputs_without_aux, targets)

                # Compute the average number of target boxes accross all nodes, for normalization purposes
                num_boxes = sum(len(t["labels"]) for t in targets)
                num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
                num_boxes = torch.clamp(num_boxes, min=1).item()

                # Compute all the requested losses
                losses = {}
                for loss in self.losses:
                    losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

                # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
                if "aux_outputs" in outputs:
                    for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                        with torch.no_grad():
                            indices = self.matcher(aux_outputs, targets)
                        for loss in self.losses:

                            l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                            l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                            losses.update(l_dict)

                return losses

        if model is None:
            model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=True)

        func_type = type(model.forward)
        model.forward = func_type(grad_enabled_forward, model)  # type: ignore

        super().__init__(
            model=model,
            clip_values=clip_values,
            channels_first=channels_first,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
            device_type=device_type,
        )

        # Check clip values
        if self.clip_values is not None:
            if not np.all(self.clip_values[0] == 0):
                raise ValueError("This estimator requires normalized input images with clip_vales=(0, 1).")
            if not np.all(self.clip_values[1] == 1):  # pragma: no cover
                raise ValueError("This estimator requires normalized input images with clip_vales=(0, 1).")

        if self.postprocessing_defences is not None:
            raise ValueError("This estimator does not support `postprocessing_defences`.")

        self._input_shape = input_shape
        cost_class = 1.0
        cost_bbox = 5.0
        cost_giou = 2.0
        bbox_loss_coef = 5.0
        giou_loss_coef = 2.0
        eos_coef = 0.1
        self.max_norm = 0.1
        num_classes = 91

        matcher = HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)
        self.weight_dict = {"loss_ce": 1, "loss_bbox": bbox_loss_coef, "loss_giou": giou_loss_coef}
        losses = ["labels", "boxes", "cardinality"]
        self.criterion = SetCriterion(
            num_classes, matcher=matcher, weight_dict=self.weight_dict, eos_coef=eos_coef, losses=losses
        )

        # Set device
        self._device: torch.device
        if device_type == "cpu" or not torch.cuda.is_available():
            self._device = torch.device("cpu")
        else:  # pragma: no cover
            cuda_idx = torch.cuda.current_device()
            self._device = torch.device(f"cuda:{cuda_idx}")

        self._model.to(self._device)
        self._model.eval()
        self.attack_losses: Tuple[str, ...] = attack_losses

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
        return self._input_shape

    @property
    def device(self) -> "torch.device":
        """
        Get current used device.

        :return: Current used device.
        """
        return self._device

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
        import cv2
        import torch

        # check if image with min, max dimensions, if not scale to 1000
        # if is within min, max dims, but not square, resize to max of image
        if (
            self._input_shape[1] < self.MIN_IMAGE_SIZE
            or self._input_shape[1] > self.MAX_IMAGE_SIZE
            or self._input_shape[2] < self.MIN_IMAGE_SIZE
            or self.input_shape[2] > self.MAX_IMAGE_SIZE
        ):
            resized_imgs = []
            for i, _ in enumerate(x):
                resized_imgs.append(
                    cv2.resize(
                        (x * 255)[i].transpose(1, 2, 0).astype(np.uint8),
                        dsize=(1000, 1000),
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
            x = (np.array(resized_imgs) / 255).transpose(0, 3, 1, 2).astype(np.float32)
        elif self._input_shape[1] != self._input_shape[2]:
            rescale_dim = max(self._input_shape[1], self._input_shape[2])
            resized_imgs = []
            for i, _ in enumerate(x):
                resized_imgs.append(
                    cv2.resize(
                        (x * 255)[i].transpose(1, 2, 0).astype(np.uint8),
                        dsize=(rescale_dim, rescale_dim),
                        interpolation=cv2.INTER_CUBIC,
                    )
                )
            x = (np.array(resized_imgs) / 255).transpose(0, 3, 1, 2).astype(np.float32)

        x = x.copy()

        self._model.eval()

        # Apply preprocessing
        x_preprocessed, _ = self._apply_preprocessing(x, y=None, fit=False)

        if self.clip_values is not None:
            norm_factor = self.clip_values[1]
        else:
            norm_factor = 1.0

        x_preprocessed_tensor = torch.from_numpy(x_preprocessed).to(self.device)
        x_preprocessed_tensor /= norm_factor

        model_output = self._model(x_preprocessed_tensor)

        predictions: List[Dict[str, np.ndarray]] = []
        for i in range(x_preprocessed_tensor.shape[0]):
            predictions.append(
                {
                    "boxes": rescale_bboxes(
                        model_output["pred_boxes"][i, :, :], (self._input_shape[1], self._input_shape[2])
                    )
                    .detach()
                    .numpy(),
                    "labels": model_output["pred_logits"][i, :, :]
                    .unsqueeze(0)
                    .softmax(-1)[0, :, :-1]
                    .max(dim=1)[1]
                    .detach()
                    .numpy(),
                    "scores": model_output["pred_logits"][i, :, :]
                    .unsqueeze(0)
                    .softmax(-1)[0, :, :-1]
                    .max(dim=1)[0]
                    .detach()
                    .numpy(),
                }
            )
        return predictions

    def _get_losses(
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Tuple[Dict[str, "torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        """
        Get the loss tensor output of the model including all preprocessing.

        :param x: Samples of shape (nb_samples, nb_channels, height, width).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The fields of the Dict are as
                  follows:
                  - boxes (FloatTensor[N, 4]): the boxes in [x1, y1, x2, y2] format, with 0 <= x1 < x2 <= W and
                                               0 <= y1 < y2 <= H.
                  - labels (Int64Tensor[N]): the labels for each image
        :return: Loss gradients of the same shape as `x`.
        """
        import torch

        self._model.train()

        self.set_dropout(False)
        self.set_multihead_attention(False)

        # Apply preprocessing
        if self.all_framework_preprocessing:
            if y is not None and isinstance(y, list) and isinstance(y[0]["boxes"], np.ndarray):
                y_tensor = []
                for y_i in y:
                    y_t = {
                        "boxes": torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device),
                        "labels": torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device),
                    }
                    y_tensor.append(y_t)
            elif y is not None and isinstance(y, dict):
                y_tensor = []
                for i in range(y["boxes"].shape[0]):
                    y_t = {"boxes": y["boxes"][i], "labels": y["labels"][i]}
                    y_tensor.append(y_t)
            else:
                y_tensor = y  # type: ignore

            if isinstance(x, np.ndarray):
                if self.clip_values is not None:
                    norm_factor = self.clip_values[1]
                else:
                    norm_factor = 1.0

                x_grad = torch.from_numpy(x / norm_factor).to(self.device)
                x_grad.requires_grad = True

            else:
                x_grad = x.to(self.device)
                if x_grad.shape[2] < x_grad.shape[0] and x_grad.shape[2] < x_grad.shape[1]:
                    x_grad = torch.permute(x_grad, (2, 0, 1))

            image_tensor_list_grad = x_grad
            x_preprocessed, y_preprocessed = self._apply_preprocessing(x_grad, y=y_tensor, fit=False, no_grad=False)
            inputs_t = x_preprocessed

        elif isinstance(x, np.ndarray):
            if y is not None and isinstance(y, list) and isinstance(y[0]["boxes"], np.ndarray):
                y_tensor = []
                for y_i in y:
                    y_t = {
                        "boxes": torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device),
                        "labels": torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device),
                    }
                    y_tensor.append(y_t)
            elif y is not None and isinstance(y[0]["boxes"], np.ndarray):
                y_tensor = []
                for y_i in y_preprocessed:
                    y_t = {
                        "boxes": torch.from_numpy(y_i["boxes"]).type(torch.float).to(self.device),
                        "labels": torch.from_numpy(y_i["labels"]).type(torch.int64).to(self.device),
                    }
                    y_tensor.append(y_t)
            else:
                y_tensor = y  # type: ignore

            x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y=y_tensor, fit=False, no_grad=True)

            if self.clip_values is not None:
                norm_factor = self.clip_values[1]
            else:
                norm_factor = 1.0

            x_grad = torch.from_numpy(x_preprocessed / norm_factor).to(self.device)
            x_grad.requires_grad = True
            image_tensor_list_grad = x_grad
            inputs_t = image_tensor_list_grad

        else:
            raise NotImplementedError("Combination of inputs and preprocessing not supported.")

        outputs = self._model(inputs_t)
        loss_components = self.criterion(outputs, y_preprocessed)

        return loss_components, inputs_t, image_tensor_list_grad

    def loss_gradient(
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, "torch.Tensor"]], **kwargs
    ) -> np.ndarray:
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Samples of shape (nb_samples, nb_channels, height, width).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Tensor[N]): the predicted labels for each image
        :return: Loss gradients of the same shape as `x`.
        """
        import torch

        _y = []
        for target in y:
            cxcy_norm = revert_rescale_bboxes(
                torch.from_numpy(target["boxes"]), (self.input_shape[1], self.input_shape[2])
            )
            _y.append(
                {
                    "labels": torch.from_numpy(target["labels"]).type(torch.int64).to(self.device),
                    "boxes": cxcy_norm.to(self.device),
                    "scores": torch.from_numpy(target["scores"]).type(torch.float).to(self.device),
                }
            )

        output, inputs_t, image_tensor_list_grad = self._get_losses(x=x, y=_y)
        loss = sum(output[k] * self.weight_dict[k] for k in output.keys() if k in self.weight_dict)

        self._model.zero_grad()

        loss.backward(retain_graph=True)  # type: ignore

        if isinstance(x, np.ndarray):
            if image_tensor_list_grad.grad is not None:
                grads = image_tensor_list_grad.grad.cpu().numpy().copy()
            else:
                raise ValueError("Gradient term in PyTorch model is `None`.")
        else:
            if inputs_t.grad is not None:
                grads = inputs_t.grad.clone()
            else:
                raise ValueError("Gradient term in PyTorch model is `None`.")

        if self.clip_values is not None:
            grads = grads / self.clip_values[1]

        if not self.all_framework_preprocessing:
            grads = self._apply_preprocessing_gradient(x, grads)

        assert grads.shape == x.shape

        return grads

    def get_activations(
        self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y, batch_size: int = 128, nb_epochs: int = 20, **kwargs) -> None:
        raise NotImplementedError

    def compute_losses(
        self, x: np.ndarray, y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]]
    ) -> Dict[str, np.ndarray]:
        """
        Compute all loss components.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Dictionary of loss components.
        """
        output_tensor, _, _ = self._get_losses(x=x, y=y)
        output = {}
        for key, value in output_tensor.items():
            if key in self.attack_losses:
                output[key] = value.detach().cpu().numpy()
        return output

    def compute_loss(  # type: ignore
        self, x: Union[np.ndarray, "torch.Tensor"], y: List[Dict[str, Union[np.ndarray, "torch.Tensor"]]], **kwargs
    ) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Compute the loss of the neural network for samples `x`.

        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,
                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).
        :param y: Target values of format `List[Dict[Tensor]]`, one for each input image. The
                  fields of the Dict are as follows:

                  - boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values \
                    between 0 and H and 0 and W
                  - labels (Int64Tensor[N]): the predicted labels for each image
                  - scores (Tensor[N]): the scores or each prediction.
        :return: Loss.
        """
        import torch

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


class NestedTensor:
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/misc.py)
    """

    def __init__(self, tensors, mask: Optional["torch.Tensor"]):
        self.tensors = tensors
        self.mask = mask

    def to_device(self, device: "torch.device"):
        """
        Transfer to device
        """
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        """
        Return tensors and masks
        """
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: Union[List, "torch.Tensor"]):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/misc.py)
    """
    import torch

    # TODO make this more general
    if tensor_list[0].ndim == 3:
        # TODO make it support different-sized images
        img_shape_list = [list(img.shape) for img in tensor_list]
        max_size = img_shape_list[0]
        for sublist in img_shape_list[1:]:
            for index, item in enumerate(sublist):
                max_size[index] = max(max_size[index], item)
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        batch, _, _, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch, batch, width), dtype=torch.bool, device=device)
        for img, _, m in zip(tensor_list, tensor, mask):
            # pad_img = pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor_list, mask)


def grad_enabled_forward(self, samples: NestedTensor):
    """
    Adapted from DETR source: https://github.com/facebookresearch/detr
    (detr/models/detr.py)
    """
    import torch

    if isinstance(samples, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(samples)
    features, pos = self.backbone(samples)

    src, mask = features[-1].decompose()
    assert mask is not None
    h_s = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

    outputs_class = self.class_embed(h_s)
    outputs_coord = self.bbox_embed(h_s).sigmoid()
    out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
    if self.aux_loss:
        out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)  # pylint: disable=W0212
    return out
