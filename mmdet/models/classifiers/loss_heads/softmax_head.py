import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ...registry import HEADS
from ...utils import bias_init_with_prob


@HEADS.register_module
class SoftmaxHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels):
        super(SoftmaxHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self._init_layers()
  
    def _init_layers(self):
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_cls = nn.Conv2d(self.in_channels, self.num_classes, 1)
        self.celoss = nn.CrossEntropyLoss()    
    def init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.conv_cls(x).reshape((-1, self.num_classes))#.sigmoid()
        return x
    
    def loss(self, cls_scores, gt_labels):
        nllloss = self.celoss(cls_scores, gt_labels.squeeze())
        return dict(ce_loss=nllloss)

    def get_results(self, outs, cfg):
        labels = (outs > cfg.score_thr).cpu().numpy()
        scores = outs.cpu().detach().numpy()
        labels_list = []
        scores_list = []
        for i in range(len(labels)):
            idxes = np.where(labels[i])[0]
            labels_list.append(idxes)
            scores_list.append(np.array([scores[i, j] for j in idxes]))
        return labels_list, scores_list


             
