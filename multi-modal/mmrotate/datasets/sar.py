
# Copyright (c) OpenMMLab. All rights reserved.
from .builder import ROTATED_DATASETS
from .ssdd import SSDDDataset

@ROTATED_DATASETS.register_module()
class SARDataset(SSDDDataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""
    CLASSES = ('ship', )
    PALETTE = [
        (0, 255, 0),
    ]
