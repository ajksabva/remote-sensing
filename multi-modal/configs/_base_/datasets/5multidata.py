# dataset settings
dataset_type = 'MultidataDataset'
data_root = '' #your data_root
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(608, 608)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',  # 合并多个数据集
        datasets=[
            dict(
                type='HRSCDataset',
                ann_file=data_root + 'RGB/HRSC2016/' + 'ImageSets/trainval.txt',
                ann_subdir=data_root + 'RGB/HRSC2016/' + 'FullDataSet/Annotations/',
                img_subdir=data_root + 'RGB/HRSC2016/' + 'FullDataSet/AllImages/',
                img_prefix=data_root + 'RGB/HRSC2016/' + 'FullDataSet/AllImages/',
                pipeline=train_pipeline
            ),
            dict(
                type='ISDDDataset',
                ann_file=data_root + 'Infrared/ISDD/' + 'ImageSets/train.txt',
                ann_subdir=data_root + 'Infrared/ISDD/' + 'FullDataSet/Annotations/',
                img_subdir=data_root + 'Infrared/ISDD/' + 'FullDataSet/JPEGImages/',
                img_prefix=data_root + 'Infrared/ISDD/' + 'FullDataSet/JPEGImages/',
                pipeline=train_pipeline
            ),
            dict(
                type='SSDDDataset',
                ann_file=data_root + 'SAR/SSDD/' + 'ImageSets/train.txt',
                ann_subdir=data_root + 'SAR/SSDD/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'SAR/SSDD/' + 'FullDataSet/JPEGImages/', 
                img_prefix=data_root + 'SAR/SSDD/' + 'FullDataSet/JPEGImages/',
                pipeline=train_pipeline),
            dict(
                type='HSIDataset',
                ann_file=data_root + 'Hyperspectral/Hsi/' + 'ImageSets/trainval.txt',  
                ann_subdir=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/AllImages/', 
                img_prefix=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/AllImages/', 
                pipeline=train_pipeline),
            dict(
                type='MMshipRGBDataset',
                ann_file=data_root + 'Multispectral/MMShip/' + 'ImageSets/train.txt',
                ann_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/RGB/', 
                img_prefix=data_root + 'Multispectral/MMShip/' + 'FullDataSet/RGB/',
                pipeline=train_pipeline),
            dict(
                type='MMshipNIRDataset',
                ann_file=data_root + 'Multispectral/MMShip/' + 'ImageSets/train.txt',
                ann_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/NIR/', 
                img_prefix=data_root + 'Multispectral/MMShip/' + 'FullDataSet/NIR/',
                pipeline=train_pipeline),
        ]
    ),
    val=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='HRSCDataset',
                ann_file=data_root + 'RGB/HRSC2016/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'RGB/HRSC2016/' + 'FullDataSet/Annotations/',
                img_subdir=data_root + 'RGB/HRSC2016/' + 'FullDataSet/AllImages/',
                img_prefix=data_root + 'RGB/HRSC2016/' + 'FullDataSet/AllImages/',
                pipeline=test_pipeline
            ),
            dict(
                type='ISDDDataset',
                ann_file=data_root + 'Infrared/ISDD/' + 'ImageSets/val.txt',
                ann_subdir=data_root + 'Infrared/ISDD/' + 'FullDataSet/Annotations/',
                img_subdir=data_root + 'Infrared/ISDD/' + 'FullDataSet/JPEGImages/',
                img_prefix=data_root + 'Infrared/ISDD/' + 'FullDataSet/JPEGImages/',
                pipeline=test_pipeline
            ),
            dict(
                type='SSDDDataset',
                ann_file=data_root + 'SAR/SSDD/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'SAR/SSDD/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'SAR/SSDD/' + 'FullDataSet/JPEGImages/', 
                img_prefix=data_root + 'SAR/SSDD/' + 'FullDataSet/JPEGImages/',
                pipeline=test_pipeline),
            dict(
                type='HSIDataset',
                ann_file=data_root + 'Hyperspectral/Hsi/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/Annotations/',
                img_subdir=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/AllImages/',
                img_prefix=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/AllImages/',
                pipeline=test_pipeline),
            dict(
                type='MMshipRGBDataset',
                ann_file=data_root + 'Multispectral/MMShip/' + 'ImageSets/val.txt',
                ann_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/RGB/', 
                img_prefix=data_root + 'Multispectral/MMShip/' + 'FullDataSet/RGB/',
                pipeline=test_pipeline),
            dict(
                type='MMshipNIRDataset',
                ann_file=data_root + 'Multispectral/MMShip/' + 'ImageSets/val.txt',
                ann_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/NIR/', 
                img_prefix=data_root + 'Multispectral/MMShip/' + 'FullDataSet/NIR/',
                pipeline=test_pipeline),
        ]
    ),
    test=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type='HRSCDataset',
                ann_file=data_root + 'RGB/HRSC2016/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'RGB/HRSC2016/' + 'FullDataSet/Annotations/',
                img_subdir=data_root + 'RGB/HRSC2016/' + 'FullDataSet/AllImages/',
                img_prefix=data_root + 'RGB/HRSC2016/' + 'FullDataSet/AllImages/',
                pipeline=test_pipeline
            ),
            dict(
                type='ISDDDataset',
                ann_file=data_root + 'Infrared/ISDD/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'Infrared/ISDD/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'Infrared/ISDD/' + 'FullDataSet/JPEGImages/', 
                img_prefix=data_root + 'Infrared/ISDD/' + 'FullDataSet/JPEGImages/',
                pipeline=test_pipeline
            ),
            dict(
                type='SSDDDataset',
                ann_file=data_root + 'SAR/SSDD/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'SAR/SSDD/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'SAR/SSDD/' + 'FullDataSet/JPEGImages/', 
                img_prefix=data_root + 'SAR/SSDD/' + 'FullDataSet/JPEGImages/',
                pipeline=test_pipeline),
            dict(
                type='HSIDataset',
                ann_file=data_root + 'Hyperspectral/Hsi/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/Annotations/',
                img_subdir=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/AllImages/',
                img_prefix=data_root + 'Hyperspectral/Hsi/' + 'FullDataSet/AllImages/',
                pipeline=test_pipeline),
            dict(
                type='MMshipRGBDataset',
                ann_file=data_root + 'Multispectral/MMShip/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/RGB/', 
                img_prefix=data_root + 'Multispectral/MMShip/' + 'FullDataSet/RGB/',
                pipeline=test_pipeline),
            dict(
                type='MMshipNIRDataset',
                ann_file=data_root + 'Multispectral/MMShip/' + 'ImageSets/test.txt',
                ann_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/Annotations/', 
                img_subdir=data_root + 'Multispectral/MMShip/' + 'FullDataSet/NIR/', 
                img_prefix=data_root + 'Multispectral/MMShip/' + 'FullDataSet/NIR/',
                pipeline=test_pipeline),
        ]
    ),
)
