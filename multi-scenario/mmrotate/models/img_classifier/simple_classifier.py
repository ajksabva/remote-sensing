import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16


from ..builder import ROTATED_CLASSIFIER, build_loss


@ROTATED_CLASSIFIER.register_module()
class SimpleClassifier(BaseModule):
    def __init__(self, 
                 num_classes,
                 in_channel = 768,
                 features_dim = 256,
                 loss_cls=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')
                ):
        super(SimpleClassifier, self).__init__(init_cfg)
        self.in_channel = in_channel
        self.features_dim = features_dim
        self.num_classes = num_classes

        if loss_cls is not None:
            self.loss_cls = build_loss(loss_cls)

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channel, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, num_classes),           
        )


    
    @auto_fp16()
    def forward(self, x):
        # x: [1, 2501, 768]
        # 取第一个token [CLS] 的特征 x[:, 0, :]
        x = x[:, 0, :]
        return self.classify(x)
    

    def forward_train(self, x, scenario):
        predict = self(x)
        return dict(loss_scenario = self.loss_cls(predict, scenario))

    def simple_test(self, x):
        return F.softmax(self(x))
