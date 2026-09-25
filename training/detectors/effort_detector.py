'''
# author: DeepfakeBench Contributor
# date: 2024
# description: Class for the Effort Detector (AIGI Detection)

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building (CLIP ViT with SVD orthogonal adaptation)
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. forward: Forward-propagation

Reference:
@article{yan2024effort,
  title={Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection},
  author={Yan, Zhiyuan and Wang, Jiangming and Wang, Zhendong and Jin, Peng and
          Zhang, Ke-Yue and Chen, Shen and Yao, Taiping and Ding, Shouhong and
          Wu, Baoyuan and Yuan, Li},
  journal={arXiv preprint arXiv:2411.15633},
  year={2024}
}

Method (Section 3.2 of the paper):
- Use a Vision Foundation Model (CLIP ViT-L/14 by default) as backbone.
- For every nn.Linear in the transformer (qkv / proj / fc1 / fc2), apply SVD:
    W = U diag(S) V^T = (U_p diag(S_p) V_p^T)  +  (U_r diag(S_r) V_r^T)
  where the top-k components form the *principal* (semantic) subspace and
  the rest form the orthogonal *residual* (forgery) subspace.
- Freeze the principal subspace; train only S_r (a vector) so that the
  residual remains a strict orthogonal complement of the semantic space
  (since U_r and V_r are columns of U and V respectively, they are
  orthonormal to U_p and V_p by construction of the SVD).
- This yields ~0.19M trainable parameters on CLIP-L/14.

Memory-safety notes:
- The HF mirror endpoint is set BEFORE importing transformers (matches
  UnivFD), to avoid HuggingFace download timeouts.
- SVD is performed under torch.no_grad() and intermediate tensors are
  ``del``-ed immediately so they do not leak into the autograd graph.
- The original ``nn.Linear`` is replaced in-place; the old weight tensor
  is released as soon as the new ``OrthogonalLinear`` is constructed.
- No tensor is accumulated on ``self`` across forward calls.
'''

import os
# Use HuggingFace mirror BEFORE importing transformers
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import logging
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC

try:
    from transformers import CLIPVisionModel
except ImportError as e:  # pragma: no cover
    CLIPVisionModel = None
    _TRF_IMPORT_ERROR = e
else:
    _TRF_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# SVD-based orthogonal linear adapter
# --------------------------------------------------------------------------- #
class OrthogonalLinear(nn.Module):
    """SVD-based orthogonal weight decomposition for an ``nn.Linear``.

    Given the original weight ``W`` of shape ``[out_dim, in_dim]``, we
    factorise once at construction time::

        W = U @ diag(S) @ Vh   (full SVD, shapes: U[out,k] S[k] Vh[k,in])

    We then partition the singular components into:
      * **principal** (top ``k_p``): the semantic subspace, frozen.
      * **residual** (remaining ``k_r``): the forgery subspace,
        whose singular values ``S_r`` are the only trainable parameters.

    Because ``U_r`` and ``V_r`` are SVD columns of ``U`` and ``V``, they
    are orthonormal to ``U_p`` and ``V_p`` -- this realises the paper's
    "orthogonal modeling" property exactly, regardless of how ``S_r`` is
    optimised during training.
    """

    def __init__(self, original_linear: nn.Linear, principal_ratio: float = 0.9):
        super().__init__()
        # Move computation to CPU then back to the original device to
        # bound peak GPU memory of SVD (large for 4096x1024 etc.).
        with torch.no_grad():
            weight = original_linear.weight.detach().to(torch.float32).cpu()
            U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        out_dim, in_dim = weight.shape
        total_rank = S.size(0)
        k_p = max(1, min(total_rank - 1, int(round(total_rank * principal_ratio))))

        # Reconstruct the (frozen) principal weight as a single matrix to
        # save FLOPs at inference: W_p = U_p diag(S_p) Vh_p
        with torch.no_grad():
            W_p = (U[:, :k_p] * S[:k_p].unsqueeze(0)) @ Vh[:k_p, :]
            U_r = U[:, k_p:].contiguous()
            Vh_r = Vh[k_p:, :].contiguous()
            S_r_init = S[k_p:].clone()

        # Frozen buffers (no grad)
        self.register_buffer('W_p', W_p.to(original_linear.weight.dtype))
        self.register_buffer('U_r', U_r.to(original_linear.weight.dtype))
        self.register_buffer('Vh_r', Vh_r.to(original_linear.weight.dtype))

        # Trainable residual singular values - this is the *only* learnable
        # part of the linear (besides bias), keeping total trainable
        # parameters tiny (~0.19M for CLIP-L/14).
        self.S_r = nn.Parameter(S_r_init.to(original_linear.weight.dtype))

        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.detach().clone())
        else:
            self.register_parameter('bias', None)

        # Eagerly drop intermediate tensors so the original weight can be
        # garbage-collected once the parent module reassigns this submodule.
        del weight, U, S, Vh, W_p, U_r, Vh_r, S_r_init

        self.in_features = in_dim
        self.out_features = out_dim
        self.k_p = k_p
        self.k_r = total_rank - k_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen principal path
        out = F.linear(x, self.W_p, self.bias)
        # Trainable residual orthogonal path:
        #   x @ Vh_r^T  ->  scale by S_r  ->  @ U_r^T
        proj = x @ self.Vh_r.t()
        proj = proj * self.S_r
        out = out + proj @ self.U_r.t()
        return out

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'k_principal={self.k_p}, k_residual={self.k_r}')


def _replace_linears_with_orthogonal(module: nn.Module,
                                     target_substrings: Iterable[str],
                                     principal_ratio: float,
                                     prefix: str = '') -> int:
    """Recursively swap ``nn.Linear`` whose qualified name contains any of
    ``target_substrings`` with :class:`OrthogonalLinear`.

    Returns the number of replaced layers.
    """
    n_replaced = 0
    for name, child in list(module.named_children()):
        full_name = f'{prefix}.{name}' if prefix else name
        if isinstance(child, nn.Linear) and any(s in full_name for s in target_substrings):
            new_layer = OrthogonalLinear(child, principal_ratio=principal_ratio)
            setattr(module, name, new_layer)
            # Drop reference to the old linear so its weight can be freed.
            del child
            n_replaced += 1
        else:
            n_replaced += _replace_linears_with_orthogonal(
                child, target_substrings, principal_ratio, prefix=full_name
            )
    return n_replaced


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
@DETECTOR.register_module(module_name='effort')
class EffortDetector(AbstractDetector):
    """Effort: SVD-based orthogonal adaptation of a frozen CLIP backbone."""

    def __init__(self, config):
        super().__init__()
        if CLIPVisionModel is None:
            raise ImportError(
                'transformers is required for EffortDetector. '
                'Install it via `pip install transformers`.'
            ) from _TRF_IMPORT_ERROR

        self.config = config
        self.backbone = self.build_backbone(config)
        self.head = self.build_classifier(config)
        self.loss_func = self.build_loss(config)

    # ------------------------------------------------------------------ #
    # Builders
    # ------------------------------------------------------------------ #
    def build_backbone(self, config) -> nn.Module:
        clip_model_name = config.get('clip_model_name', 'openai/clip-vit-large-patch14')
        principal_ratio = float(config.get('principal_ratio', 0.9))
        target_substrings = config.get(
            'svd_target_substrings',
            ['q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'],
        )

        logger.info(f'Loading CLIP backbone for Effort: {clip_model_name}')
        vision_model = CLIPVisionModel.from_pretrained(clip_model_name)

        # Freeze EVERYTHING first -- the principal subspace must remain frozen.
        for p in vision_model.parameters():
            p.requires_grad = False

        # Replace target linear layers with OrthogonalLinear modules.
        n = _replace_linears_with_orthogonal(
            vision_model, target_substrings=target_substrings,
            principal_ratio=principal_ratio,
        )
        logger.info(
            f'[Effort] Replaced {n} nn.Linear layers with OrthogonalLinear '
            f'(principal_ratio={principal_ratio}, targets={target_substrings})'
        )

        # Re-enable gradients ONLY on the residual singular values of the
        # OrthogonalLinear modules (and their bias, which is also tiny).
        # W_p, U_r, Vh_r and all original CLIP weights stay frozen so the
        # principal semantic subspace is preserved.
        for m in vision_model.modules():
            if isinstance(m, OrthogonalLinear):
                m.S_r.requires_grad = True
                if m.bias is not None:
                    m.bias.requires_grad = True

        # Sanity-check: log trainable parameter count.
        n_trainable = sum(p.numel() for p in vision_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in vision_model.parameters())
        logger.info(
            f'[Effort] Backbone trainable params: {n_trainable:,} '
            f'/ total: {n_total:,} ({100.0 * n_trainable / max(1, n_total):.4f}%)'
        )
        return vision_model

    def build_classifier(self, config) -> nn.Module:
        clip_model_name = config.get('clip_model_name', 'openai/clip-vit-large-patch14')
        if 'large' in clip_model_name:
            feat_dim = 1024
        elif 'huge' in clip_model_name:
            feat_dim = 1280
        else:
            feat_dim = 768
        return nn.Linear(feat_dim, 2)

    def build_loss(self, config) -> nn.Module:
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    # ------------------------------------------------------------------ #
    # AbstractDetector required interface
    # ------------------------------------------------------------------ #
    def features(self, data_dict: dict) -> torch.Tensor:
        outputs = self.backbone(pixel_values=data_dict['image'])
        return outputs.pooler_output

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        return {'overall': self.loss_func(pred, label)}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label']
        pred = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        feat = self.features(data_dict)
        logits = self.classifier(feat)
        prob = torch.softmax(logits, dim=1)[:, 1]
        return {'cls': logits, 'prob': prob, 'feat': feat}
