"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-0706
# description: Class for the SBIDetector

Functions in the Class are summarized as:
1. __init__: Initialization
2. build_backbone: Backbone-building
3. build_loss: Loss-function-building
4. features: Feature-extraction
5. classifier: Classification
6. get_losses: Loss-computation
7. get_train_metrics: Training-metrics-computation
8. get_test_metrics: Testing-metrics-computation
9. forward: Forward-propagation

Reference:
@inproceedings{shiohara2022detecting,
  title={Detecting deepfakes with self-blended images},
  author={Shiohara, Kaede and Yamasaki, Toshihiko},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18720--18729},
  year={2022}
}
"""

import os
import logging
import datetime
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
from copy import deepcopy
from metrics.base_metrics_class import calculate_metrics_for_train

from .base_detector import AbstractDetector
from detectors import DETECTOR
from networks import BACKBONE
from loss import LOSSFUNC
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import torch.fft as fft
from typing import Union, Tuple, Dict
import logging
from training.networks.conditionGan2   import ImprovedLowFreqConditionalGenerator as LowFreqConditionalGenerator

from training.networks.conditionGan2 import ImprovedConditionalSpectralDiscriminator as SpectralDiscriminator

import math


device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)

# ==========================
class LowFreqPerturbationGenerator(nn.Module):
    """
    Wrapper for ImprovedLowFreqConditionalGenerator:
    - No random noise.
    - Condition = simple band-limited spectral descriptor from real images (2-5 cycles/pixel).
    - forward(x_fake, real_images) -> {"perturbation":..., "condition": cond_used}
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config or {}
        image_size = self.config.get('resolution', 256)
        channels = self.config.get('img_channels', 3)
        # latent_dim retains config name but becomes cond_dim
        self.latent_dim = self.config.get('latent_dim', 64)
        # descriptor spatial downsizing (r x r -> r^2 dims); choose r so r^2 ~ latent_dim or map via FC
        self.spec_out_res = int(self.config.get('spec_out_res', 8))
        self.spec_flat_dim = self.spec_out_res * self.spec_out_res

        # generator: reuse your existing ImprovedLowFreqConditionalGenerator
        # It expects (x, noise) as interface — we will pass condition vector as "noise"
        self.generator = LowFreqConditionalGenerator(image_size=image_size,
                                                             latent_dim=self.latent_dim,
                                                             channels=channels).to(device)

        # if spec_flat_dim != latent_dim, project
        if self.spec_flat_dim != self.latent_dim:
            self.spec_fc = nn.Linear(self.spec_flat_dim, self.latent_dim).to(device)
        else:
            self.spec_fc = None

        # band parameters
        self.low = self.config.get('cond_low', 2.0)   # cycles/pixel
        self.high = self.config.get('cond_high', 5.0)

    def band_descriptor_from_images(self, imgs: torch.Tensor):
        """
        Compute band-limited spectral descriptor for imgs.
        imgs: (B, C, H, W)
        Returns: (B, spec_flat_dim) in float32 on same device
        Steps:
          - FFT -> shift -> magnitude -> log1p
          - Build bandpass mask for cycles/pixel [low, high]
          - Multiply mask, channel-average, downsample to spec_out_res x spec_out_res, flatten
        """
        B, C, H, W = imgs.shape
        # compute FFT and magnitude
        fx = torch.fft.fft2(imgs)              # complex (B,C,H,W)
        fx = torch.fft.fftshift(fx)            # complex
        mag = torch.abs(fx)                    # real (B,C,H,W)
        # band mask (based on pixel distance from center)
        u = torch.arange(H, device=imgs.device).reshape(-1, 1) - H // 2
        v = torch.arange(W, device=imgs.device).reshape(1, -1) - W // 2
        d = torch.sqrt(u.float()**2 + v.float()**2)  # distances
        # convert cycles/pixel to pixel-distance: cycles/pixel * (min_dim / 2)
        min_dim = float(min(H, W))
        low_cut = self.low * (min_dim / 2.0)
        high_cut = self.high * (min_dim / 2.0)
        mask = ((d >= low_cut) & (d <= high_cut)).float().to(imgs.device)  # (H,W)
        mask = mask.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        # apply mask
        mag_masked = mag * mask  # broadcast -> (B,C,H,W)
        # channel-average
        mag_chan = mag_masked.mean(dim=1, keepdim=True)  # (B,1,H,W)
        # numerical stabilization
        mag_log = torch.log1p(mag_chan)  # (B,1,H,W)
        # downsample to (spec_out_res, spec_out_res)
        mag_small = F.interpolate(mag_log, size=(self.spec_out_res, self.spec_out_res),
                                  mode='bilinear', align_corners=False)  # (B,1,r,r)
        flat = mag_small.view(B, -1)  # (B, r*r)
        return flat

    def forward(self, x_fake: torch.Tensor) -> dict:
        """
        x_fake: (Bf, C, H, W) - source images to be perturbed
        real_images: (Br, C, H, W) - reference real images for condition (can be None)
        Returns:
            dict with keys:
                "perturbation": (Bf, C, H, W)
                "condition": (Bf, latent_dim)  --- for D usage (we use Br->mean to broadcast to Bf)
        """
        # Bf = x_fake.shape[0]
        # # compute descriptor from real_images if provided, else from x_fake (fallback)
        # if (real_images is not None) and (real_images.numel() > 0):
        #     cond_real = self.band_descriptor_from_images(real_images)  # (Br, spec_flat_dim)
        #     # map to latent_dim
        #     if self.spec_fc is not None:
        #         cond_real_mapped = self.spec_fc(cond_real)  # (Br, latent_dim)
        #     else:
        #         cond_real_mapped = cond_real  # (Br, latent_dim)
        #     # compute a batch-level target: mean over real batch
        #     cond_mean = cond_real_mapped.mean(dim=0, keepdim=True)  # (1, latent_dim)
        #     # expand to fake batch
        #     condition = cond_mean.expand(Bf, -1).contiguous()  # (Bf, latent_dim)
        # else:
        #     # fallback: compute condition from x_fake itself (per-sample)
        #     cond_fake = self.band_descriptor_from_images(x_fake)  # (Bf, spec_flat_dim)
        #     if self.spec_fc is not None:
        #         condition = self.spec_fc(cond_fake)
        #     else:
        #         condition = cond_fake  # (Bf, latent_dim)

        # # call underlying generator: forward(x_fake, condition) where generator expects (x, noise)
        perturbation = self.generator(x_fake)

        # ensure output size
        if perturbation.size(2) != x_fake.size(2) or perturbation.size(3) != x_fake.size(3):
            perturbation = F.interpolate(perturbation, size=x_fake.shape[2:], mode='bilinear', align_corners=False)

        return {
            "perturbation": perturbation,
            
        }




class SpectralDiscriminatorWrapper(nn.Module):
    """
    Wrapper that instantiates ImprovedConditionalSpectralDiscriminator and
    provides a forward(x, condition) -> (logit, feat).
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config or {}
        img_channels = self.config.get('img_channels', 3)
        img_size = self.config.get('resolution', 256)
        self.cond_dim = self.config.get('cond_dim', self.config.get('latent_dim', 64))

        self.discriminator = SpectralDiscriminator(channels=img_channels, image_size=img_size).to(device)

    def forward(self, x: torch.Tensor):
        # condition: (B, cond_dim) or None
        return self.discriminator(x)


# ============================
# 3) FrepddCgDetector 修改 forward 使用 real 的 band descriptor condition
#    替换原来的 generate_perturbation 调用与 D 调用
# ============================

@DETECTOR.register_module(module_name="frepddCg")
class FrepddCgDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.generator = LowFreqPerturbationGenerator(config.get('generator', {})).to(device)
        self.discriminator = SpectralDiscriminatorWrapper(config.get('discriminator', {})).to(device)

        # rest same as before
        self.criterion = nn.BCELoss()
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)

        self.lambda_cls = config.get('lambda_cls', 1.0)
        self.lambda_d = config.get('lambda_d', 1.0)
        self.lambda_g = config.get('lambda_g', 1.0)

    # keep your mmd_loss_spectrum, get_losses, get_d_loss, get_g_loss (unchanged) ...
    # (copy your previous implementations; I've kept them in your project)
    # only forward() is adjusted to compute condition from real_images and pass to generator & discriminator


    def generate_perturbation(self, x_fake) -> Dict:
        """生成扰动 G(x) — 已包含 noise"""
        return self.generator( x_fake)
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """判别信号真伪 D(x, condition)"""
        return self.discriminator(x)
    
    def build_backbone(self, config):
        backbone_class = BACKBONE[config["backbone_name"]]
        model_config = config["backbone_config"]
        backbone = backbone_class(model_config)
        state_dict = torch.load(config["pretrained"])
        for name, weights in state_dict.items():
            if "pointwise" in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
        backbone.load_state_dict(state_dict, False)
        logger.info("Load pretrained model successfully!")
        return backbone

    def build_loss(self, config):
        loss_class = LOSSFUNC[config["loss_func"]]
        loss_func = loss_class()
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        return self.backbone.features(data_dict["image"])

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)


    def mmd_loss_spectrum(self, x, y, low=2, high=5, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        B1, C, H, W = x.shape
        B2 = y.shape[0]

        # --- 1. FFT ---
        fx = torch.fft.fftshift(torch.fft.fft2(x))
        fy = torch.fft.fftshift(torch.fft.fft2(y))

        # 取幅度，使其成为 real tensor
        fx = torch.abs(fx)
        fy = torch.abs(fy)

        # mask
        u = torch.arange(H, device=x.device).reshape(-1, 1) - H // 2
        v = torch.arange(W, device=x.device).reshape(1, -1) - W // 2
        d = torch.sqrt(u ** 2 + v ** 2)

        low_cut = low * (min(H, W) / 2)
        high_cut = high * (min(H, W) / 2)
        mask = ((d >= low_cut) & (d <= high_cut)).float().unsqueeze(0).unsqueeze(0)

        fx = (fx * mask).reshape(B1, -1)
        fy = (fy * mask).reshape(B2, -1)

        # --- 3. kernel ---
        def gaussian_kernel(a, b, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
            total_a = a.unsqueeze(1)  # (B1,1,D)
            total_b = b.unsqueeze(0)  # (1,B2,D)
            L2 = ((total_a - total_b) ** 2).sum(2)

            # 限制 L2 范围（避免爆炸）
            L2 = torch.clamp(L2, 0, 1e6)

            # 带宽
            if fix_sigma:
                bandwidth = fix_sigma
            else:
                bandwidth = torch.mean(L2.detach())
                bandwidth = torch.clamp(bandwidth, 1e-6, 1e6)

            # 多尺度带宽
            bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

            kernel_val = 0
            for bw in bandwidth_list:
                exp_term = torch.clamp(-L2 / bw, min=-50, max=0)   # 防溢出
                kernel_val += torch.exp(exp_term)

            return kernel_val  # shape (B1,B2)

        XX = gaussian_kernel(fx, fx, kernel_mul, kernel_num, fix_sigma)
        YY = gaussian_kernel(fy, fy, kernel_mul, kernel_num, fix_sigma)
        XY = gaussian_kernel(fx, fy, kernel_mul, kernel_num, fix_sigma)

        # MMD
        return XX.mean() + YY.mean() - 2 * XY.mean()

    def get_losses(self, data_dict: dict, pred: dict) -> dict:
        cls_loss = self.get_cls_loss(data_dict, pred)
        d_loss = self.get_d_loss(data_dict, pred)
        g_loss = self.get_g_loss(data_dict, pred)
        
        overall_loss = (self.lambda_cls * cls_loss + 
                       self.lambda_d * d_loss + 
                       self.lambda_g * g_loss)
        
        return {
            "overall": overall_loss,
            "cls_loss": cls_loss,
            "d_loss": d_loss,
            "g_loss": g_loss
        }

    def get_d_loss(self, data_dict: dict, pred: dict) -> torch.Tensor:
        real_data = pred["real_data"]
        freq_data_dict = pred['freq_data_dict']
        device_ = data_dict["label"].device

        d_loss = torch.tensor(0.0, device=device_)
        valid = 0

        # Hinge loss helper
        def hinge_d_loss(real_logits, fake_logits):
            loss_real = torch.mean(F.relu(1.0 - real_logits))
            loss_fake = torch.mean(F.relu(1.0 + fake_logits))
            return loss_real + loss_fake

        # R1 gradient penalty (optional)
        r1_gamma = self.config.get('r1_gamma', 0.0)

        if real_data["image"].numel() > 0 and real_data["outputs"] is not None:
            real_logits, real_feats = real_data["outputs"]  # logits, feat
            # convert to shape (B,)
            real_logits = real_logits.view(real_logits.size(0))
            # we need fake logits for hinge; get from freq_data_dict if available
            if freq_data_dict["outputsd"] is not None:
                fake_logits, fake_featsd = freq_data_dict["outputsd"]
                fake_logits = fake_logits.view(fake_logits.size(0))
            else:
                fake_logits = torch.zeros_like(real_logits)

            d_loss = hinge_d_loss(real_logits, fake_logits)
            valid += 1

        else:
            # If no real images, still compute fake-only hinge (use zeros for real logits)
            if freq_data_dict["outputsd"] is not None:
                fake_logits, _ = freq_data_dict["outputsd"]
                fake_logits = fake_logits.view(fake_logits.size(0))
                d_loss = torch.mean(F.relu(1.0 + fake_logits))
                valid += 1

        if valid > 0:
            # nothing to average here beyond hinge
            pass

        return d_loss

    def get_g_loss(self, data_dict: dict, pred: dict) -> torch.Tensor:
        
        x = data_dict["image"]
        labels = data_dict["label"]
        
        fake_mask = labels == 1
        fake_images = x[fake_mask]
        fake_labels = labels[fake_mask]
        
        freq = pred['freq_data_dict']
        fake_imgs = freq["image"]            
        
        real_imgs = pred["real_data"]["image"]

        # 获取 fake logits（用于 adversarial loss）
        if freq["outputs"] is None:
            return torch.tensor(0.0, device=fake_imgs.device)

        fake_logits, _ = freq["outputs"]
        fake_logits = fake_logits.view(fake_logits.size(0))

        # adversarial hinge loss
        g_adv = -torch.mean(fake_logits)

        # ----- 1. Reconstruction loss -----
        lambda_rec = self.config.get("lambda_rec", 10.0)  # 推荐 10~50
        rec_loss = F.l1_loss(fake_imgs, fake_images)

        # ----- 2. MMD loss (spectral-domain) -----
        lambda_mmd = self.config.get("lambda_mmd", 1.0)  # 推荐 1~5

        if real_imgs.numel() > 0:
            mmd = self.mmd_loss_spectrum(fake_imgs, real_imgs)
        else:
            mmd = torch.tensor(0.0, device=fake_imgs.device)

        # ----- Final G loss -----
        g_loss = g_adv + lambda_mmd * mmd + lambda_rec * rec_loss
        return g_loss

    def get_cls_loss(self, data_dict: dict, pred: dict) -> torch.Tensor:
        pred_dict = pred['pred_dict']
        if pred_dict["cls"] is not None:
            label = data_dict["label"]
            cls_pred = pred_dict["cls"]
            cls_loss = self.loss_func(cls_pred, label)
            return cls_loss
        
        return torch.tensor(0.0, device=data_dict["label"].device)
  
    def forward(self, data_dict: dict, inference=False, for_generator=False, for_discriminator=False, for_classifier=False) -> dict:
        x = data_dict["image"]
        labels = data_dict["label"]

        # split
        real_mask = labels == 0
        real_images = x[real_mask]
        real_labels = labels[real_mask]

        fake_mask = labels == 1
        fake_images = x[fake_mask]
        fake_labels = labels[fake_mask]

        if inference:
            with torch.no_grad():
                # generate using available fake_images and real_images as reference if possible
                gen_out = self.generate_perturbation(fake_images)
                perturbation = gen_out["perturbation"]
                # for inference you might want to return concatenated set
                perturbation = torch.cat([real_images, perturbation], dim=0) if real_images.numel()>0 else perturbation
                features = self.features({"image": perturbation})
                pred = self.classifier(features)
                prob = torch.softmax(pred, dim=1)[:, 1]
                scores = torch.softmax(pred, dim=1)
            return {
                "pred_dict": {"cls": pred, "prob": prob, "feat": features, "scores": scores},
                "perturbation": perturbation,
                "x_fake":fake_images,
                "x_real":real_images,
            }
        print(fake_images.shape, real_images.shape)
        # 2. generate perturbation using real_images as condition if present
        gen_out = self.generate_perturbation(fake_images)
        perturbation = gen_out["perturbation"]
        condition = gen_out.get("condition", None)  # (Bf, cond_dim) ; cond is broadcasted inside D

        perturbed_images = perturbation  # this is the "fake" images fed to D and used for classifier

        # 3. classifier forward (if requested)
        if for_classifier:
            concat_imgs = torch.cat([real_images, perturbed_images], dim=0) if real_images.numel()>0 else perturbed_images
            features = self.features({"image": concat_imgs})
            pred = self.classifier(features)
            prob = torch.softmax(pred, dim=1)[:, 1]
            scores = torch.softmax(pred, dim=1)
        else:
            features = pred = prob = scores = None

        # 4. discriminator forward
        real_outputs = real_feats = None
        fake_outputs = fake_feats = None
        fake_outputsd = fake_featsd = None

        if for_discriminator or for_generator:
            # real: compute per-sample condition (descriptor) to pair with each real
            real_condition = None
            if real_images.numel() > 0:
                # compute real per-sample descriptor to feed D with (one-to-one)
                real_cond = self.generator.band_descriptor_from_images(real_images)  # (Br, spec_flat_dim)
                if self.generator.spec_fc is not None:
                    real_cond = self.generator.spec_fc(real_cond)  # (Br, cond_dim)
                # D on real images with their own condition
                real_outputs, real_feats = self.discriminate(real_images)

            # fake: use condition returned by generator (which was real-batch-mean expanded to Bf)
            if perturbed_images.numel() > 0:
                fake_outputs, fake_feats = self.discriminate(perturbed_images)
                # detach version for D training
                if for_discriminator:
                    cond_detach = None if condition is None else condition.detach()
                    fake_outputsd, fake_featsd = self.discriminate(perturbed_images.detach())

        result = {
            "real_data": {"image": real_images, "label": real_labels, "outputs": (real_outputs, real_feats)},
            "freq_data_dict": {
                "image": perturbed_images,
                "label": fake_labels,
                "outputs": (fake_outputs, fake_feats),
                "outputsd": (fake_outputsd, fake_featsd)
            },
            "pred_dict": {"cls": pred, "prob": prob, "feat": features, "scores": scores}
        }

        return result

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
        self.video_names = []
        return metric_batch_dict

