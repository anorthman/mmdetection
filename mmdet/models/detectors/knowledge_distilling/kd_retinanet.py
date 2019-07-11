# author huangchuanhong
from .kd_single_stage import KDSingleStageDetector
from ...registry import DETECTORS


@DETECTORS.register_module
class KDRetinaNet(KDSingleStageDetector):

    def __init__(self,
                 backbone,
                 teacher,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(KDRetinaNet, self).__init__(backbone, teacher, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
