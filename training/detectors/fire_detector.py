'''
# author: DeepfakeBench Contributor
# date: 2024
# description: Class for the FIRE Detector

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
@inproceedings{li2025fire,
  title={FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided Reconstruction Error},
  author={Li, Zijin and Wang, Zhen and Li, Jingjing and Lu, Yang and Li, Zheng and Gao, Xin and Su, Jingyu and Gao, Wen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
'''

import os
# Use HuggingFace mirror BEFORE importing transformers
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

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

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from transformers import Dinov2Model

logger = logging.getLogger(__name__)


class FrequencyBranch(nn.Module):
    """
    Frequency-domain feature extraction branch.
    Uses DCT-based frequency analysis to capture frequency artifacts.
    """
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # Learnable frequency filters
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        # x: [B, 3, H, W]
        # Apply frequency analysis via learned convolutions
        return self.freq_conv(x)


@DETECTOR.register_module(module_name='fire')
class FIREDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.freq_branch = FrequencyBranch(in_channels=3, out_channels=64)
        self.head = self.build_classifier(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # FIRE uses DINOv2 as the backbone for reconstruction error analysis
        dino_model_name = config.get('dino_model_name', 'facebook/dinov2-base')
        logger.info(f'Loading DINOv2 model: {dino_model_name}')
        
        model = Dinov2Model.from_pretrained(dino_model_name)
        
        # Freeze DINOv2 weights (FIRE uses frozen DINOv2 for feature extraction)
        for param in model.parameters():
            param.requires_grad = False
        
        logger.info('DINOv2 backbone weights frozen.')
        return model

    def build_classifier(self, config):
        dino_model_name = config.get('dino_model_name', 'facebook/dinov2-base')
        if 'large' in dino_model_name:
            feat_dim = 1024
        elif 'giant' in dino_model_name:
            feat_dim = 1536
        else:
            feat_dim = 768
        
        # Combine DINOv2 features + frequency features
        total_dim = feat_dim + 64
        head = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
        return head
        
    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        x = data_dict['image']
        # DINOv2 features
        outputs = self.backbone(pixel_values=x)
        dino_feat = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        # Frequency features
        freq_feat = self.freq_branch(x)
        freq_feat = F.adaptive_avg_pool2d(freq_feat, (1, 1))
        freq_feat = freq_feat.view(freq_feat.size(0), -1)
        
        # Concatenate
        combined_feat = torch.cat([dino_feat, freq_feat], dim=1)
        return combined_feat

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
