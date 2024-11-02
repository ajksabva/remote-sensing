# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
from mmdet.datasets.pipelines import LoadImageFromFile, LoadAnnotations

from ..builder import ROTATED_PIPELINES



@ROTATED_PIPELINES.register_module()
class LoadPatchFromImage(LoadImageFromFile):
    """Load an patch from the huge image.

    Similar with :obj:`LoadImageFromFile`, but only reserve a patch of
    ``results['img']`` according to ``results['win']``.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with image in ``results['img']``.

        Returns:
            dict: The dict contains the loaded patch and meta information.
        """

        img = results['img']
        x_start, y_start, x_stop, y_stop = results['win']
        width = x_stop - x_start
        height = y_stop - y_start

        patch = img[y_start:y_stop, x_start:x_stop]
        if height > patch.shape[0] or width > patch.shape[1]:
            patch = mmcv.impad(patch, shape=(height, width))

        if self.to_float32:
            patch = patch.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = patch
        results['img_shape'] = patch.shape
        results['ori_shape'] = patch.shape
        results['img_fields'] = ['img']
        return results

@ROTATED_PIPELINES.register_module()
class LoadAnnotationsWithScenario(LoadAnnotations):
    def __call__(self, results):
        results = super(LoadAnnotationsWithScenario, self).__call__(results)
        results = self._load_img_cls(results)
        return results
    

    def _load_img_cls(self, results):
        ann_info = results['ann_info']
        results['scenario'] = ann_info['scenario'].copy()
        return results