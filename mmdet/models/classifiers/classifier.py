import cv2
import torch.nn as nn
import logging
import copy

from .. import builder
from ..registry import CLASSIFIERS

from mmdet.core import tensor2imgs

@CLASSIFIERS.register_module
class Classifier(nn.Module):
   
    def __init__(self,
                 backbone,
                 loss_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Classifier, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.loss_head = builder.build_head(loss_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from {}'.format(pretrained))
        self.backbone.init_weights(pretrained=pretrained)
        self.loss_head.init_weights()

    def forward_train(self, img, label, **kwargs):
        x = self.backbone(img)
        if isinstance(x, list):
            x = x[0]
        outs = self.loss_head(x)
        losses = self.loss_head.loss(outs, label)
        return losses 

    def forward_test(self, imgs, labels, **kwargs):
        x = self.backbone(imgs)
        if isinstance(x, list):
            x = x[0]
        outs = self.loss_head(x)
        labels, scores = self.loss_head.get_results(outs, self.test_cfg)
        return labels, scores

    def forward(self, img, label=None, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, label, **kwargs)
        else:
            return self.forward_test(img, label, **kwargs)
       
    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    classes):
        img_tensor = data['img']
        img_norm_cfg_255 = copy.deepcopy(img_norm_cfg)
        img_norm_cfg_255['mean'] = [x * 255. for x in img_norm_cfg['mean']]
        img_norm_cfg_255['std'] = [x * 255. for x in img_norm_cfg['std']]
        img = tensor2imgs(img_tensor,**img_norm_cfg_255)[0]
        img = cv2.resize(img, (256, 256))
        labels_list, scores_list = result
        labels = labels_list[0]
        scores = scores_list[0]
        for i in range(len(labels)):
            idx = labels[i]
            img = cv2.putText(img, classes[idx] + ': %.2f'%(scores[i]), (20, 20 + i * 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
        cv2.imshow('img', img)
        cv2.waitKey(0)
