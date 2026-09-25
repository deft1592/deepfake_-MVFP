"""ImageNet-pretrained ViT backbone for DeepfakeBench classifiers."""

import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from metrics.registry import BACKBONE


logger = logging.getLogger(__name__)


@BACKBONE.register_module(module_name="vit_base_patch16_224")
class ViTBasePatch16(nn.Module):
    """ViT-B/16 adapter exposing DeepfakeBench's features/classifier API."""

    loads_own_pretrained = True

    def __init__(self, config):
        super().__init__()
        self.image_size = int(config.get("image_size", 224))
        self.num_classes = int(config.get("num_classes", 2))
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=1000,
        )

        pretrained_path = config.get("pretrained")
        if pretrained_path:
            if not os.path.isfile(pretrained_path):
                raise FileNotFoundError(
                    f"ViT pretrained weights do not exist: {pretrained_path}"
                )
            state_dict = torch.load(pretrained_path, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            clean_state_dict = {}
            for key, value in state_dict.items():
                clean_key = key[7:] if key.startswith("module.") else key
                if not clean_key.startswith("head."):
                    clean_state_dict[clean_key] = value
            state_dict = clean_state_dict
            incompatible = self.vit.load_state_dict(state_dict, strict=False)
            expected_missing = {"head.weight", "head.bias"}
            unexpected_missing = set(incompatible.missing_keys) - expected_missing
            if unexpected_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "ViT pretrained weights are incompatible: "
                    f"missing={sorted(unexpected_missing)}, "
                    f"unexpected={incompatible.unexpected_keys}"
                )
            logger.info("Loaded ImageNet ViT weights from %s", pretrained_path)

        # The ImageNet head is intentionally replaced for binary classification.
        self.vit.head = nn.Identity()
        dropout = float(config.get("dropout", 0.0))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.last_layer = nn.Linear(self.vit.num_features, self.num_classes)
        nn.init.trunc_normal_(self.last_layer.weight, std=0.02)
        nn.init.zeros_(self.last_layer.bias)

    def features(self, x):
        # FREPDD still operates at 256x256; only its classifier sees 224x224.
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return self.vit.forward_features(x)

    def classifier(self, features):
        # timm 0.6 returns all tokens while newer versions may return pooled features.
        embedding = features[:, 0] if features.ndim == 3 else features
        self.last_emb = embedding
        return self.last_layer(self.dropout(embedding))

    def forward(self, x):
        return self.classifier(self.features(x))
