from mmdet.datasets import CustomDataset
from .builder import ROTATED_DATASETS

@ROTATED_DATASETS.register_module()
class MultidataDataset(CustomDataset):
    CLASSES = ()  # 没有类别

    def __init__(self,
                 ann_file,
                 pipeline,
                 img_subdir='JPEGImages',
                 ann_subdir='Annotations',
                 classwise=False,
                 version='oc',
                 **kwargs):
        # 初始化空数据集
        super(MultidataDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_file):
        # 返回一个空列表，表示没有数据
        return []

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
        
        return []
