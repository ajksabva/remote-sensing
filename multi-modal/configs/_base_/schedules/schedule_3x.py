# evaluation
evaluation = dict(interval=1, metric='mAP', save_best='auto')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3, 
    step=[24, 33]) #表示在第 24 个 epoch 和第 33 个 epoch 时，学习率将下降。具体的下降方式通常是将当前学习率乘以一个预定义的衰减因子
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=20)
