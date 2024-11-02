import os
import os.path as osp
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict


import mmcv
from mmcv import print_log
import numpy as np
import torch
from mmdet.datasets.custom import CustomDataset
from terminaltables import AsciiTable

from mmrotate.core import obb2poly_np, poly2obb_np
from mmrotate.core.evaluation import eval_rbbox_map, eval_scenario
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class MASATIv2Dataset(CustomDataset):

    CLASSES = ('ship',)
    SCENARIO_CLASSES = ('coast_ship', 'ship', 'coast', 'land', 'water')
    SCENARIO_CLASSES_FILTER = ('coast_ship', 'ship', 'land')

    def __init__(self,
                 ann_file,
                 pipeline,
                 version="le90",
                 img_subdir = osp.join('FullDataSet', 'AllImages'),
                 ann_subdir = osp.join('FullDataSet', 'Annotations'),
                 img_prefix = osp.join('FullDataSet', 'AllImages'),
                 filter_imgs = False,
                 **kwargs):
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.img_prefix = img_prefix
        self.filter_imgs = filter_imgs
        self.version = version
        if filter_imgs:
            self.SCENARIO_CLASSES = self.SCENARIO_CLASSES_FILTER

        super(MASATIv2Dataset, self).__init__(ann_file, pipeline, **kwargs)
    
    def __len__(self):
        return len(self.data_infos)
    
    def load_annotations(self, ann_file):
        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)   

        for img_id in img_ids:
            data_info = {}       
            filename = osp.join(self.img_subdir, f'{img_id}.bmp')
            data_info['filename'] = filename
            ann_path = osp.join(self.ann_subdir, f'{img_id}.xml')

            tree = ET.parse(ann_path)
            root = tree.getroot()       

            width = int(root.find('Img_SizeWidth').text)
            height = int(root.find('Img_SizeHeight').text)

            data_info['width'] = width
            data_info['height'] = height

            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            
            objs = root.findall('HRSC_Objects/HRSC_Object')
            scenario = root.find('Img_Class').text
            if self.filter_imgs and scenario not in self.SCENARIO_CLASSES:
                continue
    
            scenario = self.scenario_str2int(scenario)

            for obj in objs:
                label = 0
                bbox = np.array([[
                    float(obj.find('mbox_cx').text),
                    float(obj.find('mbox_cy').text),
                    float(obj.find('mbox_w').text),
                    float(obj.find('mbox_h').text),
                    float(obj.find('mbox_ang').text),
                    0
                ]], 
                    dtype=np.float32)
                # polygon: np.array[x0, y0, x1, y0, x1, y1, x0, y1]
                polygon = obb2poly_np(bbox, 'le90')[0, :-1].astype(np.float32)
                
                if self.version != 'le90':
                    bbox = np.array(
                        poly2obb_np(polygon, self.version), dtype=np.float32)
                else:
                    # bbox: np.array[cx, cy, w, h, ang]
                    bbox = bbox[0, :-1]
                gt_bboxes.append(bbox)
                gt_labels.append(label)
                gt_polygons.append(polygon)
            

            if gt_bboxes:
                data_info['ann'] = dict(
                    scenario = np.array(scenario),
                    bboxes = np.array(gt_bboxes, dtype=np.float32),
                    labels = np.array(gt_labels, dtype=np.int64),
                    polygons = np.array(gt_polygons, dtype=np.float32)
                )
            else:
                data_info['ann'] = dict(
                    scenario = np.array(scenario),
                    bboxes = np.zeros((0,5), dtype=np.float32),
                    labels = np.zeros((0,), dtype=np.int64),
                    polygons = np.zeros((0,8), dtype=np.float32)
                )
            data_infos.append(data_info)
        
        return data_infos

    def _filter_imgs(self):
        valid_inds = range(len(self.data_infos))
        return valid_inds

    def evaluate(
            self,
            results,
            metric='mAP',
            logger=None,
            proposal_nums=(100, 300, 1000),
            iou_thrs=[0.5],
            scale_ranges=None,
            use_07_metric=True,
            nproc=4
    ):
        
        scenario_results = [r['scenario'] for r in results]
        results = [r['bbox'] for r in results]

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        assert isinstance(iou_thrs, list)

        eval_scenario(scenario_results, annotations, self.SCENARIO_CLASSES)


        mean_aps = []
        eval_results = OrderedDict()
        for iou_thr in iou_thrs:
            print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            mean_ap, _ = eval_rbbox_map(
                results,
                annotations,
                scale_ranges=scale_ranges,
                iou_thr=iou_thr,
                use_07_metric=use_07_metric,
                nproc=nproc
            )
            mean_aps.append(mean_ap)
            eval_results[f'AP{int(iou_thr*100):02d}'] = round(mean_ap, 3)
        eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
        eval_results.move_to_end('mAP', last=False)
        return eval_results
            
    def scenario_str2int(self, str):
        return self.SCENARIO_CLASSES.index(str)
    
    

    


