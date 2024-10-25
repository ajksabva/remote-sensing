# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv import print_log
from mmdet.datasets import CustomDataset
from PIL import Image

from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class SSDDDataset(CustomDataset):
    
    CLASSES = ('ship', )
    def __init__(self,
                 ann_file,
                 pipeline,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 classwise=False,
                 version='oc',
                 **kwargs):
        self.img_subdir = img_subdir
        self.ann_subdir = ann_subdir
        self.classwise = classwise
        self.version = version
        # load_annotations
        super(SSDDDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotation from XML style ann_file.

        Args:
            ann_file (str): Path of Imageset file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = mmcv.list_from_file(ann_file)
        for img_id in img_ids:
            data_info = {}

            filename = osp.join(self.img_subdir, f'{img_id}.jpg')
            data_info['filename'] = f'{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, self.ann_subdir,
                                f'{img_id}.xml')
            
            tree = ET.parse(xml_path)
            root = tree.getroot()

            width = int(root.find('size/width').text)
            height = int(root.find('size/height').text)

            if width is None or height is None:
                img_path = osp.join(self.img_prefix, filename)
                img = Image.open(img_path)
                width, height = img.size
            data_info['width'] = width
            data_info['height'] = height
            data_info['ann'] = {}
            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_headers = []

            # 循环遍历每个图片中的所有物体
            for obj in root.findall('object'):
                label = 0
                # Add an extra score to use obb2poly_np
                # bbox shape(1,6)
                bbox = np.array([[
                    float(obj.find('rotated_bndbox/rotated_bbox_cx').text),
                    float(obj.find('rotated_bndbox/rotated_bbox_cy').text),
                    float(obj.find('rotated_bndbox/rotated_bbox_w').text),
                    float(obj.find('rotated_bndbox/rotated_bbox_h').text),
                    float(obj.find('rotated_bndbox/rotated_bbox_theta').text), 0
                ]],
                                dtype=np.float32)
                #将旋转边界框 (Oriented Bounding Box, OBB) 转换为多边形 (Polygon) 的顶点坐标。
                polygon = obb2poly_np(bbox, 'le90')[0, :-1].astype(np.float32) 
                if self.version != 'le90':
                    bbox = np.array(
                        poly2obb_np(polygon, self.version), dtype=np.float32)
                else:
                    bbox = bbox[0, :-1]  #bbox shape(1,5) 去掉了类别
                # 表示目标的头部坐标,标注信息中没有头部坐标，用检测框的中心点来代替。
                head = np.array([
                    # int((float(obj.find('rotated_bndbox/x1').text) + float(obj.find('rotated_bndbox/x2').text) + float(obj.find('rotated_bndbox/x3').text) + float(obj.find('rotated_bndbox/x4').text)) / 4),
                    # int((float(obj.find('rotated_bndbox/y1').text) + float(obj.find('rotated_bndbox/y2').text) + float(obj.find('rotated_bndbox/y3').text) + float(obj.find('rotated_bndbox/y4').text)) / 4)
                    int(obj.find('rotated_bndbox/rotated_bbox_cx').text),
                    int(obj.find('rotated_bndbox/rotated_bbox_cy').text)
                ],
                                dtype=np.int64)

                gt_bboxes.append(bbox)
                gt_labels.append(label)
                gt_polygons.append(polygon)
                gt_headers.append(head)

            #如果有标记信息
            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(
                    gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(
                    gt_polygons, dtype=np.float32)
                data_info['ann']['headers'] = np.array(
                    gt_headers, dtype=np.int64)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5), dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8),
                                                        dtype=np.float32)
                data_info['ann']['headers'] = np.zeros((0, 2),
                                                       dtype=np.float32)
            
            data_info['ann']['bboxes_ignore'] = np.zeros((0, 5),
                                                            dtype=np.float32)
            data_info['ann']['labels_ignore'] = np.array([],
                                                            dtype=np.int64)
            data_info['ann']['polygons_ignore'] = np.zeros(
                (0, 8), dtype=np.float32)
            data_info['ann']['headers_ignore'] = np.zeros((0, 2),
                                                            dtype=np.float32)

            data_infos.append(data_info)
        return data_infos

    def evaluate(
            self,
            results,
            metric='mAP',
            logger=None,
            proposal_nums=(100, 300, 1000),
            iou_thr=[0.5],
            scale_ranges=None,
            use_07_metric=True,
            nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            use_07_metric (bool): Whether to use the voc07 metric.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            raise NotImplementedError

        return eval_results
