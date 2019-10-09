import logging
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch.nn as nn

class BaseClassifier(nn.Module):
    """Base class for classifier"""
    
    __metaclass__ = ABCMeta

    def __init__(self):
        super(BaseClassifier, self).__init__()
       
    @abstractmethod
    def forward_train(self, imgs, labels, **kwargs):
        pass

    def forward_test(self, imgs, labels, **kwargs):
        pass

    def forward(self, img, label, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(img, label, **kwargs)
        else:
            return self.forward_test(img, label, **kwargs)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
