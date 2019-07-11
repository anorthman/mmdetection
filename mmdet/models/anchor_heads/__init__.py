from .anchor_head import AnchorHead
from .rpn_head import RPNHead
from .retina_head import RetinaHead
from .ssd_head import SSDHead
from .knowledge_distilling import KDRetinaHead

__all__ = ['AnchorHead', 'RPNHead', 'RetinaHead', 'SSDHead', 'KDRetinaHead']
