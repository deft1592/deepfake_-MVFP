"""Backbones required by the AAAI27 FREPDD-CLIP reproduction."""

import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import BACKBONE
from .clip_vit import CLIPViTLargePatch14, CLIPViTBasePatch16

__all__ = ["BACKBONE", "CLIPViTLargePatch14", "CLIPViTBasePatch16"]

