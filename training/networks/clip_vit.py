"""OpenAI CLIP vision backbone adapter for DeepfakeBench classifiers."""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel

from metrics.registry import BACKBONE


logger = logging.getLogger(__name__)


@BACKBONE.register_module(module_name="clip_vit_large_patch14")
class CLIPViTLargePatch14(nn.Module):
    """CLIP ViT-L/14 exposing DeepfakeBench's features/classifier API."""

    loads_own_pretrained = True

    def __init__(self, config):
        super().__init__()
        pretrained_path = config.get("pretrained")
        if not pretrained_path or not os.path.isdir(pretrained_path):
            raise FileNotFoundError(
                f"CLIP pretrained model directory does not exist: {pretrained_path}"
            )

        self.image_size = int(config.get("image_size", 224))
        self.num_classes = int(config.get("num_classes", 2))
        self.clip = CLIPVisionModel.from_pretrained(
            pretrained_path,
            local_files_only=True,
        )
        if config.get("gradient_checkpointing", True):
            self.clip.gradient_checkpointing_enable()

        hidden_size = int(self.clip.config.hidden_size)
        dropout = float(config.get("dropout", 0.0))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.last_layer = nn.Linear(hidden_size, self.num_classes)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)
        nn.init.zeros_(self.last_layer.bias)
        logger.info(
            "Loaded CLIP vision backbone %s from %s",
            self.clip.config.model_type,
            pretrained_path,
        )

    def features(self, x):
        # FREPDD remains at 256x256; only the CLIP classifier sees 224x224.
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return self.clip(pixel_values=x).pooler_output

    def classifier(self, features):
        self.last_emb = features
        return self.last_layer(self.dropout(features))

    def forward(self, x):
        return self.classifier(self.features(x))


@BACKBONE.register_module(module_name="clip_vit_base_patch16")
class CLIPViTBasePatch16(CLIPViTLargePatch14):
    """CLIP ViT-B/16 exposing DeepfakeBench's features/classifier API."""
