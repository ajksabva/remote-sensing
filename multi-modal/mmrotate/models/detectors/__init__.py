# Copyright (c) OpenMMLab. All rights reserved.
from .base import RotatedBaseDetector
from .two_stage import RotatedTwoStageDetector
from .oriented_rcnn import OrientedRCNN

__all__ = [
    'RotatedBaseDetector', 'RotatedTwoStageDetector', 
    'RotatedimTED', 'OrientedRCNN',
]
