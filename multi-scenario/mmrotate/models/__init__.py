'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-10-25 09:09:47
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-10-27 06:46:28
FilePath: /remote-sensing/multi-scenario/mmrotate/models/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) OpenMMLab. All rights reserved.
from .backbones import *  # noqa: F401, F403
from .builder import (ROTATED_BACKBONES, ROTATED_DETECTORS, ROTATED_HEADS,
                      ROTATED_LOSSES, ROTATED_NECKS, ROTATED_ROI_EXTRACTORS,
                      ROTATED_SHARED_HEADS, ROTATED_CLASSIFIER,
                      build_backbone, build_detector,
                      build_head, build_loss, build_neck, build_roi_extractor,
                      build_shared_head, build_img_classifier)
from .dense_heads import *  # noqa: F401, F403
from .detectors import *  # noqa: F401, F403
from .losses import *  # noqa: F401, F403
from .necks import *  # noqa: F401, F403
from .roi_heads import *  # noqa: F401, F403
from .utils import *  # noqa: F401, F403
from .img_classifier import *

__all__ = [
    'ROTATED_BACKBONES', 'ROTATED_NECKS', 'ROTATED_ROI_EXTRACTORS',
    'ROTATED_SHARED_HEADS', 'ROTATED_HEADS', 'ROTATED_LOSSES',
    'ROTATED_DETECTORS', 'ROTATED_CLASSIFIER',
    'build_backbone', 'build_neck', 'build_roi_extractor',
    'build_shared_head', 'build_head', 'build_loss', 'build_detector',
    'build_img_classifier'
]
