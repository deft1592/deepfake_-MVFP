'''
# author: DeepfakeBench Contributor
# date: 2024
# description: Class for the FreqBlender Detector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building (EfficientNet-B4 with advprop)
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. forward: Forward-propagation

Reference:
@inproceedings{freqblender2024,
  title={FreqBlender: Enhancing DeepFake Detection by Blending Frequency Knowledge},
  author={Wang, Hanzhe and Liu, Yangyang and Zhao, Hao and Lyu, Siwei and Li, Yan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024}
}

Notes on memory safety:
- The detector is stateless across calls (no Python list accumulators
  keeping tensors alive across batches), avoiding the unbounded growth
  pattern present in some other detectors (e.g. SBIDetector).
- ``pred_dict`` returns the bare ``features`` tensor without retaining
  any reference inside ``self``; consumers are responsible for detaching
  or moving to CPU before storage.
- Backbone weights for EfficientNet-B4 are loaded once via
  ``efficientnet_pytorch`` and reused. Optional checkpoint loading
  releases temporary state dicts immediately after ``load_state_dict``.
'''

import os
import logging
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

try:
    from efficientnet_pytorch import EfficientNet
except ImportError as e:  # pragma: no cover
    EfficientNet = None
    _EFFNET_IMPORT_ERROR = e
else:
    _EFFNET_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='freqblender')
class FreqBlenderDetector(AbstractDetector):
    """FreqBlender's downstream classifier.

    The original FreqBlender pipeline trains a frequency-aware data
    augmentation module and finally fits an EfficientNet-B4 (advprop)
    binary classifier on the augmented samples. This detector wraps that
    EfficientNet-B4 so that DeepfakeBench can both train the classifier
    end-to-end and evaluate the released ``FreqBlender.tar`` checkpoint
    against arbitrary (LMDB or raw) test datasets.
    """

    def __init__(self, config):
        super().__init__()
        if EfficientNet is None:
            raise ImportError(
                "efficientnet_pytorch is required for FreqBlenderDetector. "
                "Install it via `pip install efficientnet_pytorch`."
            ) from _EFFNET_IMPORT_ERROR

        self.config = config
        self.num_classes = int(config.get('num_classes', 2))
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        # Optional FreqBlender released checkpoint (the EfficientNet-B4 head
        # already trained on FreqBlender-augmented data).
        ckpt_path = config.get('freqblender_ckpt', None)
        if ckpt_path and os.path.isfile(ckpt_path):
            self._load_freqblender_checkpoint(ckpt_path)

    # ------------------------------------------------------------------ #
    # Building blocks
    # ------------------------------------------------------------------ #
    def build_backbone(self, config):
        """Build EfficientNet-B4 backbone exactly like FreqBlender's
        ``inference/model.py``: ``advprop=True`` + ``num_classes=2``.
        """
        backbone_weight = config.get('backbone_weight', None)
        weights_path = backbone_weight if (backbone_weight and os.path.isfile(backbone_weight)) else None

        advprop = bool(config.get('advprop', True))
        if weights_path is not None:
            logger.info(f'Loading EfficientNet-B4 backbone from local file: {weights_path}')
            net = EfficientNet.from_pretrained(
                'efficientnet-b4',
                weights_path=weights_path,
                advprop=advprop,
                num_classes=self.num_classes,
            )
        else:
            # Try to use online pretrained weights; fall back to random init.
            try:
                logger.info('Loading EfficientNet-B4 backbone with advprop pretrained weights.')
                net = EfficientNet.from_pretrained(
                    'efficientnet-b4',
                    advprop=advprop,
                    num_classes=self.num_classes,
                )
            except Exception as exc:
                logger.warning(
                    f'Failed to download pretrained EfficientNet-B4 ({exc}); '
                    f'initializing from scratch.'
                )
                net = EfficientNet.from_name(
                    'efficientnet-b4',
                    num_classes=self.num_classes,
                )
        return net

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    def _load_freqblender_checkpoint(self, ckpt_path: str) -> None:
        """Load the released FreqBlender classifier checkpoint.

        The published ``FreqBlender.tar`` stores ``{"model": state_dict}``
        whose keys start with ``net.`` (matching FreqBlender's
        ``Detector(nn.Module)`` whose only attribute is ``self.net``).
        We strip that prefix so weights map directly onto our
        ``self.backbone`` (an ``EfficientNet`` instance).

        We immediately release the intermediate dictionaries so that
        residual references do not keep CUDA tensors alive after load.
        """
        logger.info(f'Loading FreqBlender released checkpoint: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt

        # Strip ``net.`` prefix coming from FreqBlender's wrapper module.
        new_sd = {}
        for k, v in state_dict.items():
            new_k = k[len('net.'):] if k.startswith('net.') else k
            new_sd[new_k] = v

        missing, unexpected = self.backbone.load_state_dict(new_sd, strict=False)
        if missing:
            logger.warning(f'FreqBlender ckpt missing keys ({len(missing)}): e.g. {missing[:3]}')
        if unexpected:
            logger.warning(f'FreqBlender ckpt unexpected keys ({len(unexpected)}): e.g. {unexpected[:3]}')

        # Free the temporary copies aggressively to avoid lingering references.
        del state_dict, new_sd, ckpt

    # ------------------------------------------------------------------ #
    # AbstractDetector required interface
    # ------------------------------------------------------------------ #
    def features(self, data_dict: dict) -> torch.Tensor:
        """Extract penultimate features (after global pooling, before fc)."""
        x = data_dict['image']
        # ``EfficientNet.extract_features`` returns spatial feature map.
        feat_map = self.backbone.extract_features(x)
        feat = F.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        # ``feat_map`` is no longer needed; let it be garbage-collected.
        del feat_map
        return feat

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        # ``EfficientNet`` keeps its head as ``_dropout`` + ``_fc``.
        # ``features`` here are post-GAP, so we replicate the same path.
        if hasattr(self.backbone, '_dropout') and self.backbone._dropout is not None:
            features = self.backbone._dropout(features)
        return self.backbone._fc(features)

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        # Run the backbone once: extract features + classifier head.
        # This mirrors EfficientNet's own ``forward`` while exposing the
        # post-GAP feature vector to DeepfakeBench utilities.
        feat = self.features(data_dict)
        logits = self.classifier(feat)
        prob = torch.softmax(logits, dim=1)[:, 1]
        return {'cls': logits, 'prob': prob, 'feat': feat}

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        loss = self.loss_func(pred, label)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
