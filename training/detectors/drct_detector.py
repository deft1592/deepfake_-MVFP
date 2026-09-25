'''
# author: DeepfakeBench Contributor
# date: 2024
# description: Class for the DRCT Detector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. forward: Forward-propagation

Reference:
@inproceedings{chen2024drct,
  title={DRCT: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images},
  author={Chen, Zhiyuan and Zhang, Bingdong and Li, Yao and Zheng, Xiaozhong and Zou, Xiang and Yao, Yiming and Ren, Aiguo},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
'''

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from torchvision import models

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='drct')
class DRCTDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = self.build_classifier(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # DRCT uses ConvNeXt or ResNet as detector backbone
        # Here we provide ConvNeXt-base as default, matching the paper's best config
        backbone_name = config.get('backbone_name', 'convnext_base')
        logger.info(f'Building DRCT backbone: {backbone_name}')
        
        if backbone_name == 'convnext_base':
            backbone = models.convnext_base(pretrained=True)
            # Replace classifier with Identity
            backbone.classifier[-1] = nn.Identity()
            feat_dim = 1024
        elif backbone_name == 'convnext_small':
            backbone = models.convnext_small(pretrained=True)
            backbone.classifier[-1] = nn.Identity()
            feat_dim = 768
        elif backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            backbone.fc = nn.Identity()
            feat_dim = 2048
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented for DRCT")
        
        self._feat_dim = feat_dim
        return backbone

    def build_classifier(self, config):
        head = nn.Linear(self._feat_dim, 2)
        return head
        
    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        x = self.backbone.features(data_dict['image']) if hasattr(self.backbone, 'features') else self.backbone(data_dict['image'])
        # Global average pooling for ConvNeXt
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(x.size(0), -1)
        return x

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.head(features)
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        loss_dict = {'overall': loss}
        return loss_dict
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        features = self.features(data_dict)
        pred = self.classifier(features)
        prob = torch.softmax(pred, dim=1)[:, 1]
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict
