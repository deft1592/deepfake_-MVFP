'''
# author: DeepfakeBench Contributor
# date: 2024
# description: Class for the UnivFD Detector

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
@inproceedings{ojha2023towards,
  title={Towards Universal Fake Image Detectors that Generalize Across Generative Models},
  author={Ojha, Utkarsh and Li, Yong Jae and Lee, Yong Jae},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={24480--24489},
  year={2023}
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
from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='univfd')
class UnivFDDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = self.build_classifier(config)
        self.loss_func = self.build_loss(config)
        
    def build_backbone(self, config):
        # prepare the backbone
        clip_model_name = config.get('clip_model_name', 'openai/clip-vit-large-patch14')
        logger.info(f'Loading CLIP model: {clip_model_name}')
        
        # Load CLIP vision model
        vision_model = CLIPVisionModel.from_pretrained(clip_model_name)
        
        # Freeze CLIP backbone weights (UnivFD keeps CLIP frozen)
        for param in vision_model.parameters():
            param.requires_grad = False
        
        logger.info('CLIP backbone weights frozen.')
        return vision_model

    def build_classifier(self, config):
        # Get feature dimension based on CLIP model
        clip_model_name = config.get('clip_model_name', 'openai/clip-vit-large-patch14')
        if 'large' in clip_model_name:
            feat_dim = 1024
        else:
            feat_dim = 768
        
        # Simple linear classifier as in UnivFD paper
        head = nn.Linear(feat_dim, 2)
        return head
        
    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config['loss_func']]
        loss_func = loss_class()
        return loss_func
    
    def features(self, data_dict: dict) -> torch.tensor:
        # Extract CLIP visual features
        outputs = self.backbone(pixel_values=data_dict['image'])
        # Use pooler_output (CLS token representation)
        feat = outputs.pooler_output
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
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
        return metric_batch_dict
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        # get the features by backbone
        features = self.features(data_dict)
        # get the prediction by classifier
        pred = self.classifier(features)
        # get the probability of the pred
        prob = torch.softmax(pred, dim=1)[:, 1]
        # build the prediction dict for each output
        pred_dict = {'cls': pred, 'prob': prob, 'feat': features}
        return pred_dict
