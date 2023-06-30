# Copyright 2020 - present, Facebook, Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utility methods for DEtection TRansformer (DETR) in PyTorch.

 | Paper link: https://arxiv.org/abs/2005.12872

 Changes/differences to original code:
 - Line 209: add device
 - Line 243: remove reference to box_ops import
 - Line 327: remove check for distributed computing
 - Line 391: add device
 - Lines 456-7: remove copy_()
 - Line 459: returning original tensor list
 - Line 462: function name changed to distinguish that it now facilitates gradients
"""

from typing import List, Optional, Tuple, Union
import torch


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
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


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

        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = torch.nn.functional.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight.to(src_logits.device)
        )
        losses = {"loss_ce": loss_ce}
        return losses

    @staticmethod
    def loss_cardinality(outputs, targets):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = torch.nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = torch.nn.functional.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)))
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
        target_masks, _ = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to_device(src_masks)
        target_masks = target_masks[tgt_idx]

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

        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        """
        permute targets following indices
        """

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


def box_cxcywh_to_xyxy(x: "torch.Tensor"):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/box_ops.py)
    """

    x_c, y_c, width, height = x.unbind(1)
    box = [(x_c - 0.5 * width), (y_c - 0.5 * height), (x_c + 0.5 * width), (y_c + 0.5 * height)]
    return torch.stack(box, dim=1)


def box_xyxy_to_cxcywh(x: "torch.Tensor"):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/box_ops.py)
    """

    x_0, y_0, x_1, y_1 = x.unbind(-1)
    box = [(x_0 + x_1) / 2, (y_0 + y_1) / 2, (x_1 - x_0), (y_1 - y_0)]
    return torch.stack(box, dim=-1)


def rescale_bboxes(out_bbox: "torch.Tensor", size: Tuple[int, int]):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (inference notebook)
    """

    img_w, img_h = size
    box = box_cxcywh_to_xyxy(out_bbox)
    box = box * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return box


def revert_rescale_bboxes(out_bbox: "torch.Tensor", size: Tuple[int, int]):
    """
    Adapted from DETR source: https://github.com/facebookresearch/detr
    (inference notebook)

    This method reverts bounding box rescaling to match input image size
    """

    img_w, img_h = size
    box = out_bbox / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox.device)
    box = box_xyxy_to_cxcywh(box)
    return box


def box_iou(boxes1: "torch.Tensor", boxes2: "torch.Tensor"):
    """
    From DETR source: https://github.com/facebookresearch/detr
    (detr/util/box_ops.py)
    """
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

    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    l_t = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    r_b = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    w_h = (r_b - l_t).clamp(min=0)  # [N,M,2]
    area = w_h[:, :, 0] * w_h[:, :, 1]

    return iou - (area - union) / area


def nested_tensor_from_tensor_list(tensor_list: Union[List, "torch.Tensor"]):
    """
    Adapted from DETR source: https://github.com/facebookresearch/detr
    (detr/util/misc.py)
    """

    if tensor_list[0].ndim == 3:
        img_shape_list = [list(img.shape) for img in tensor_list]
        max_size = img_shape_list[0]
        for sublist in img_shape_list[1:]:
            for index, item in enumerate(sublist):
                max_size[index] = max(max_size[index], item)
        batch_shape = [len(tensor_list)] + max_size
        batch, _, _, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch, batch, width), dtype=torch.bool, device=device)
        for img, _, m in zip(tensor_list, tensor, mask):
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor_list, mask)


def grad_enabled_forward(self, samples: NestedTensor):
    """
    Adapted from DETR source: https://github.com/facebookresearch/detr
    (detr/models/detr.py)
    """

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
