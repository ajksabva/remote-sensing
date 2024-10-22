'''
Author: Guosy_wxy 1579528809@qq.com
Date: 2024-10-22 10:02:17
LastEditors: Guosy_wxy 1579528809@qq.com
LastEditTime: 2024-10-22 11:05:06
FilePath: /Spatial-Transform-Decoupling-final/code/mmcv_custom/__init__.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-

from .checkpoint import load_checkpoint
from .layer_decay_optimizer_constructor import LayerDecayOptimizerConstructor




__all__ = ['load_checkpoint', 'LayerDecayOptimizerConstructor']
