# author huangchuanhong
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (AnchorGenerator, anchor_target, delta2bbox,
                        multi_apply, weighted_cross_entropy, weighted_smoothl1,
                        weighted_binary_cross_entropy,
                        weighted_sigmoid_focal_loss,
                        weighted_sigmoid_kldiv_focal_loss,
                        weighted_sigmoid_kldiv, 
                        multiclass_nms, attention_loss)
from ...registry import HEADS

from ..retina_head import RetinaHead

@HEADS.register_module
class KDRetinaHead(RetinaHead):
    '''
    add kd_loss to loss, implemented only for retinanet
    '''
    def loss_single(self, cls_score, bbox_pred, teacher_cls_scores,
                    teacher_bbox_preds, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss 
        if self.use_sigmoid_cls:
            labels = labels.reshape(-1, self.cls_out_channels)
            label_weights = label_weights.reshape(-1, self.cls_out_channels)
        else:
            raise NotImplementedError
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        teacher_cls_scores = teacher_cls_scores.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        if self.use_sigmoid_cls:
            if self.use_focal_loss:
                cls_criterion = weighted_sigmoid_focal_loss
                #if cfg.teacher.kd_with_focal:
                #    kd_cls_criterion = weighted_sigmoid_kldiv_focal_loss
                #else:
                #    kd_cls_criterion = weighted_sigmoid_kldiv
            else:
                raise NotImplementedError
        else:
            if self.use_focal_loss:
                raise NotImplementedError
            else:
                raise NotImplementedError
        if self.use_focal_loss:
            if cfg.focal_loss == 0.:
                loss_cls = torch.zeros([]).cuda() 
            else:
                loss_cls = cls_criterion(
                    cls_score,
                    labels,
                    label_weights,
                    gamma=cfg.gamma,
                    alpha=cfg.alpha,
                    avg_factor=num_total_samples)
            if cfg.teacher.kd_with_focal:
                loss_kd_cls = weighted_sigmoid_kldiv_focal_loss(
                    cls_score,
                    teacher_cls_scores,
                    label_weights,
                    temperature=cfg.teacher.temperature,
                    gamma=cfg.gamma,
                    alpha=cfg.alpha,
                    teacher_alpha=cfg.teacher.alpha,
                    avg_factor=num_total_samples)
            else:
                loss_kd_cls = weighted_sigmoid_kldiv(
                    cls_score,
                    teacher_cls_scores,
                    label_weights,
                    temperature=cfg.teacher.temperature,
                    teacher_alpha=cfg.teacher.alpha,
                    avg_factor=num_total_samples)
        else:
            raise NotImplementedError
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
        loss_reg = weighted_smoothl1(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            beta=cfg.smoothl1_beta,
            avg_factor=num_total_samples)
        return loss_cls, loss_kd_cls, loss_reg

    def loss(self,
             features,
             cls_scores,
             bbox_preds,
             teacher_features,
             teacher_cls_scores,
             teacher_bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas)
        sampling = False if self.use_focal_loss else True
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (num_total_pos if self.use_focal_loss else
                             num_total_pos + num_total_neg)
        losses_cls, losses_kd_cls, losses_reg = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            teacher_cls_scores,
            teacher_bbox_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg)
        losses_at, = multi_apply(
            attention_loss,
            features,
            teacher_features,
            beta=cfg.teacher.beta)
        return dict(loss_cls=losses_cls, loss_kd_cls=losses_kd_cls, loss_reg=losses_reg,
                    loss_at=losses_at)
