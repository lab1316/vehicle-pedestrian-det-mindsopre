# import torch
import math
# from torch import nn
import copy

# from general_config.general_config import device
from general_config.anchor_config import default_boxes
from general_config import constants
from utils.preprocessing import map_id_to_idx
import numpy as np
import mindspore as ms
from mindspore import ops,nn

# inspired by fastai http://course18.fast.ai/lessons/lesson9.html course


class Classification_Loss(nn.Cell):
    def __init__(self, params):
        super().__init__()
        self.loss_type = params.loss_type
        self.focal_loss = params.use_focal_loss

    def construct(self, pred, targ):
        """
        Arguments:
            pred - tensor of shape batch x n_classes x anchors
            targ - tensor of shape batch x anchors

        Explanation:
            computes softmax/BCE loss between model prediction and target
            model predicts scores for each class
            for BCE the ground truth is represented as a one-hot matrix
            for softmax it is just a list of indeces
        Returns: softmax loss or (weighted if focal) BCE loss
        """
        batch, n_classes, n_anchors = pred.shape
        class_idx = map_id_to_idx(targ)

        if self.loss_type == constants.BCE_loss:
            pred = pred.permute(0, 2, 1).contiguous()
            one_hot = ops.one_hot(class_idx.astype(ms.int64), depth=n_classes+1).astype(ms.float32)
            # one_hot = one_hot.to(device)

            # remove background column
            one_hot = one_hot[:, :, :-1]

            weight = self.get_weight(pred, one_hot) if self.focal_loss else None
            bce_loss = ops.binary_cross_entropy_with_logits(pred, one_hot,
                                                                            weight=weight,
                                                                            reduction='none')
            return bce_loss.sum(axis=2)
        else:
            return ops.cross_entropy(pred, class_idx, reduction='none')

    def get_weight(self, x, t):
        # focal loss decreases loss for correctly classified (P>0.5) examples, relative to the missclassified ones
        # thus increasing focus on them
        alpha, gamma = 0.25, 2.
        p = x.sigmoid()

        # focal loss factor - decreases relative loss for well classified examples
        pt = p*t + (1-p)*(1-t)

        # counter positive/negative examples imbalance by assigning higher relative values to positives=1
        w = alpha*t + (1-alpha)*(1-t)

        # these two combined strongly encourage the network to predict a high value when
        # there is indeed a positive example
        return w * ((1-pt).pow(gamma))

    def map_id_to_idx(self, class_ids):
        """
        maps the tensor of class ids to indeces
        """
        class_idx = ops.zeros(class_ids.shape, dtype=ms.int32)
        for k, v in self.id2idx.items():
            class_idx[class_ids == k] = v

        # class_idx = class_idx.to(device)
        return class_idx


class Detection_Loss():
    """
    Computes both localization and classification loss
    """

    def __init__(self, params):
        self.anchors_xywh = default_boxes(order="xywh")
        # self.anchors_xywh = self.anchors_xywh.to(device)

        self.params = params
        self.hard_negative = params.use_hard_negative_mining
        self.class_loss = Classification_Loss(self.params)

        self.scale_xy = 10
        self.scale_wh = 5

        self.anchors_batch = self.anchors_xywh.unsqueeze(dim=0)

    def ssd_loss(self, pred, targ):
        """
        Arguments:
            pred - model output - two tensors of dim B x 4 x #anchors and B x n_classes x #anchors in a list
            targ - ground truth - two tensors of dim B x #anchors x 4 and B x #anchors in a list

        Explanation:
        each image loss is normalized by the number of anchors to obj mappings
        total loss is normalized by the batch size

        Return: loc and class loss per whole batch
        """
        pred_bbox, pred_id = pred[0].permute(0, 2, 1), pred[1]
        gt_bbox, gt_id = targ

        # compute offsets
        offsets = self.prepare_localization_offsets(gt_bbox)

        pos_mask = gt_id != 100
        pos_num = pos_mask.sum(axis=1)

        # B x 1
        localization_loss = self.localization_loss(pos_mask, pred_bbox, offsets)
        # B x 1
        classification_loss = self.classification_loss(pos_mask, pos_num, pred_id, gt_id)

        # normalize by mappings per each image in the batch then take the mean
        # we skip images without annotations, so no element in pos_num is 0
        localization_loss = (localization_loss / pos_num.float()).mean(axis=0)
        classification_loss = (classification_loss / pos_num.float()).mean(axis=0)
        return localization_loss, classification_loss

    def hard_negative_mining(self, pos_mask, pos_num, losses, ids_for_anchors, ratio=3):
        """
        Taken from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
        """
        tem = ms.Tensor(np.zeros(losses.shape))
        tem = losses
        losses_ = copy.deepcopy(tem)
        losses_[pos_mask] = -math.inf
        _, indexes = losses_.sort(axis=1, descending=True)
        _, orders = indexes.sort(axis=1)
        neg_num = ops.clamp(ratio*pos_num, max=pos_mask.shape[1]).unsqueeze(-1)
        neg_mask = orders < neg_num
        return pos_mask | neg_mask

    def localization_loss(self, pos_mask, pred_bbox, offsets):
        """
        Arguments:
        pos_mask - indeces of matched anchors
        pred_bbox - [B x #anchors x 4] tensor - model predictions
        offsets - [B x #anchors x 4] tensor - ground truth

        returns: l1 loss between predictions and ground truth batch_bboxes per each image in batch
        """
        # loss for each anchor
        loc_loss = (ops.smooth_l1_loss(
            pred_bbox, offsets, reduction='none')).sum(axis=2)

        # return the loss for those that actually matched
        return (loc_loss * pos_mask.float()).sum(axis=1)

    def classification_loss(self, pos_mask, pos_num, pred_id, gt_id):
        """
        Arguments:
        pos_mask - indeces of matched anchors
        pos_num - how many mappings for each image
        pred_id - [batch x n_classes x #anchors] tensor - confidence scores by each anchor
        gt_id - [batch x #anchors x 1] tensor - ground truth class ids

        returns: softmax/BCE between predicted scores and gt for each image in batch
        """
        class_losses = self.class_loss(pred_id, gt_id)
        if self.hard_negative:
            mask = self.hard_negative_mining(pos_mask, pos_num, class_losses, gt_id)
            loss = (class_losses * mask.float()).sum(axis=1)
        else:
            loss = class_losses.sum(axis=1)

        return loss

    def prepare_localization_offsets(self, gt_bbox):
        """
        Arguments:
        - gt_bbox - B x #anchors x 4 tensor - ground truth bounding boxes

        returns - offsets
        """
        off_xy = self.scale_xy*(gt_bbox[:, :, :2] -
                                self.anchors_batch[:, :, :2])/self.anchors_batch[:, :, 2:]
        off_wh = self.scale_wh*(gt_bbox[:, :, 2:]/self.anchors_batch[:, :, 2:]).log()
        return ops.cat((off_xy, off_wh), axis=2).contiguous()
