from .anchor_head import AnchorHead
from .fcos_head import FCOSHead
from .ga_retina_head import GARetinaHead
from .ga_rpn_head import GARPNHead
from .guided_anchor_head import FeatureAdaption, GuidedAnchorHead
from .reppoints_head import RepPointsHead
from .retina_head import RetinaHead
from .rpn_head import RPNHead
from .ssd_head import SSDHead
from .solo_head import SoloHead
# from .solo_head_vec import SoloHead
__all__ = [
    'AnchorHead', 'GuidedAnchorHead', 'FeatureAdaption', 'RPNHead',
    'GARPNHead', 'RetinaHead', 'GARetinaHead', 'SSDHead', 'FCOSHead',
    'RepPointsHead','SoloHead'
]
