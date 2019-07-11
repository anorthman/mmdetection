# author huangchuanhong
import torch
from mmcv.runner import load_checkpoint
from ..base import BaseDetector
from ..single_stage import SingleStageDetector
from ...registry import DETECTORS
from ...builder import build_detector


@DETECTORS.register_module
class KDSingleStageDetector(SingleStageDetector):
    def __init__(self,
                 backbone,
                 teacher,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KDSingleStageDetector, self).__init__(backbone,
                                                    neck=neck,
                                                    bbox_head=bbox_head,
                                                    train_cfg=train_cfg,
                                                    test_cfg=test_cfg,
                                                    pretrained=pretrained)
        self.teacher_detector = build_detector(teacher.model, train_cfg=None, test_cfg=test_cfg)
        load_checkpoint(self.teacher_detector, teacher.checkpoint)
        self.teacher_detector.eval()
        self.beta = train_cfg.teacher.beta

    def forward_train(self,
                     img,
                     img_metas,
                     gt_bboxes,
                     gt_labels,
                     gt_bboxes_ignore=None,
                     beta=1000.):
        feats = ()
        backbone_feats = self.backbone(img)
        if self.train_cfg.teacher.backbone_at:
            for i in self.train_cfg.teacher.backbone_at_idxes:
                feats += (backbone_feats[i],)
        if self.with_neck:
            neck_feats = self.neck(backbone_feats)
            if self.train_cfg.teacher.neck_at:
                feats += neck_feats
            outs = self.bbox_head(neck_feats)
        else:
            outs = self.bbox_head(backbone_feats)
        with torch.no_grad():
            t_feats = ()
            t_backbone_feats = self.teacher_detector.backbone(img)
            if self.train_cfg.teacher.backbone_at:
                for i in self.train_cfg.teacher.backbone_at_idxes:
                    t_feats += (t_backbone_feats[i],)
            if self.with_neck:
                t_neck_feats = self.teacher_detector.neck(t_backbone_feats)
                if self.train_cfg.teacher.neck_at:
                    t_feats += t_neck_feats
                t_outs = self.teacher_detector.bbox_head(t_neck_feats)
            else:
                t_outs = self.teacher_detector.bbox_head(t_backbone_feats)
        loss_inputs = (feats,) + outs + (t_feats,) + t_outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg) 
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses
