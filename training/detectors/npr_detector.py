'''
# author: DeepfakeBench Contributor
# date: 2024
# description: Class for the NPR Detector

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
@inproceedings{tan2024rethinking,
  title={Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection},
  author={Tan, Chuangchuang and Zhao, Yao and Wei, Bin and Ni, Ronggang and Sun, Zhaoxia and Huang, Jing and Zhao, Shijie and Liu, Sam Kwong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={26618--26628},
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


class HighPassFilter(nn.Module):
    """
    High-pass filter for noise pattern extraction.
    Uses fixed 3x3 kernels inspired by SRM (Spatial Rich Model) filters.
    """
    def __init__(self, in_channels=3):
        super().__init__()
        # Multiple high-pass kernels for noise extraction
        # Kernel 1: Laplacian-like
        # Kernel 2: Horizontal edge
        # Kernel 3: Vertical edge
        kernels = torch.tensor([
            [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]],
            [[0., 0., 0.], [0., -1., 0.], [0., 0., 0.]],
            [[1., 2., 1.], [2., -12., 2.], [1., 2., 1.]],  # Laplacian
            [[-1., 2., -1.], [2., -4., 2.], [-1., 2., -1.]],  # High-pass
        ], dtype=torch.float32).unsqueeze(1)  # [4, 1, 3, 3]
        
        # Replicate for RGB channels
        self.register_buffer('kernels', kernels)
        self.in_channels = in_channels
        self.out_channels = 4 * in_channels
        
    def forward(self, x):
        # x: [B, 3, H, W]
        outputs = []
        for c in range(self.in_channels):
            for k in range(self.kernels.size(0)):
                kernel = self.kernels[k:k+1, ...]  # [1, 1, 3, 3]
                out = F.conv2d(x[:, c:c+1, :, :], kernel, padding=1)
                outputs.append(out)
        return torch.cat(outputs, dim=1)  # [B, 12, H, W]


@DETECTOR.register_module(module_name='npr')
class NPRDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hpf = HighPassFilter(in_channels=3)
        self.backbone = self.build_backbone(config)
        self.head = self.build_classifier(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # Use ResNet50 as backbone following NPR paper
        backbone_name = config.get('backbone_name', 'resnet50')
        logger.info(f'Building NPR backbone: {backbone_name}')
        
        if backbone_name == 'resnet50':
            backbone = models.resnet50(pretrained=True)
            # Modify first conv to accept 12 channels (4 filters * 3 RGB)
            original_conv = backbone.conv1
            backbone.conv1 = nn.Conv2d(
                12, original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            # Initialize new conv with averaged weights
            with torch.no_grad():
                backbone.conv1.weight[:, :3, :, :] = original_conv.weight
                backbone.conv1.weight[:, 3:6, :, :] = original_conv.weight
                backbone.conv1.weight[:, 6:9, :, :] = original_conv.weight
                backbone.conv1.weight[:, 9:12, :, :] = original_conv.weight
            
            # Remove final fc layer
            backbone.fc = nn.Identity()
        else:
            raise NotImplementedError(f"Backbone {backbone_name} not implemented for NPR")
        
        return backbone

    def build_classifier(self, config):
        feat_dim = 2048  # ResNet50 output dim
        head = nn.Linear(feat_dim, 2)
        return head
        
    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        x = data_dict['image']
        # Apply high-pass filter
        residual = self.hpf(x)
        # Extract features
        feat = self.backbone(residual)
        return feat

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
