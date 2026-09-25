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

class LowFreqPerturbationGenerator(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        image_size = config.get('resolution', 256)
        channels = config.get('img_channels', 3)
        
        # 简单频谱描述子维度（例如 8×8=64）
        self.spec_dim = config.get('spec_dim', 64)

        # LFC Generator 输入不再包含 noise
        self.generator = LowFreqConditionalGenerator(
            image_size=image_size,
            cond_dim=self.spec_dim,      # 频谱描述子作为条件
            img_channels=channels
        ).to(device)

    # -------- 重点：简单频谱描述子 ----------
    def simple_spectral_descriptor(self, x, out_res=8):
        """
        x: (B, C, H, W)
        输出: (B, out_res*out_res)
        """
        B, C, H, W = x.shape
        
        # 1) FFT 幅度谱（real tensor）
        fx = torch.fft.fft2(x)
        fx = torch.abs(torch.fft.fftshift(fx))  # (B,C,H,W)

        # 2) 通道平均 → (B,1,H,W)
        fx = fx.mean(dim=1, keepdim=True)

        # 3) 下采样到 out_res × out_res
        fx_small = F.interpolate(fx, size=(out_res, out_res), mode="bilinear", align_corners=False)

        # 4) 展开 → 频谱向量
        return fx_small.reshape(B, -1)  # (B, out_res*out_res)


    # -------- forward ---------
    def forward(self, x: torch.Tensor) -> Dict:
        """
        输入:
            x: fake 图像 (B,C,H,W)
        输出:
            perturbation: 生成图
            condition: 频谱描述子 (B, spec_dim)
        """

        # 获取频谱描述子
        c_spec = self.simple_spectral_descriptor(x)  # (B, spec_dim)

        # G(x, c_spec)
        perturbation = self.generator(x, c_spec)

        if perturbation.size(2) != x.size(2):
            perturbation = F.interpolate(perturbation, size=x.shape[2:], mode='bilinear')

        return {
            "perturbation": perturbation,
            "condition": c_spec  # 返回给判别器使用
        }
    

class SpectralDiscriminatorWrapper(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()

        img_channels = config.get('img_channels', 3)
        img_size = config.get('resolution', 128)

        # 与 Generator 相同
        self.cond_dim = config.get('spec_dim', 64)

        self.discriminator = SpectralDiscriminator(
            img_channels=img_channels,
            cond_dim=self.cond_dim,
            image_size=img_size
        ).to(device)

    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        return self.discriminator(x, condition)


    # 修改 get_losses 不再直接调用 sigmoid; 保留 for compatibility but not used with hinge
    def get_losses(self, data_dict: Dict, pred: torch.Tensor) -> torch.Tensor:
        label = data_dict['label'].float().unsqueeze(1).to(pred[0].device)
        # If pred is (logit, feat)
        if isinstance(pred, tuple):
            logit = pred[0]
        else:
            logit = pred
        prob = torch.sigmoid(logit)
        return F.binary_cross_entropy(prob, label) 
    def _initialize_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)
    
    def init_weights(self, pretrained_path: Union[bool, str] = None):
        if pretrained_path is None:
            self._initialize_weights()
        elif isinstance(pretrained_path, str):
            self.load_state_dict(torch.load(pretrained_path))


# 3) FrepddCgDetector（关键修改：判别器调用使用 noise 作为 condition）
@DETECTOR.register_module(module_name="frepddCg")
class FrepddCgDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0
        
        # 创建生成器和判别器（传入各自 config）
        self.generator = LowFreqPerturbationGenerator(
            config.get('generator', {})
        )
        
        self.discriminator = SpectralDiscriminatorWrapper(
            config.get('discriminator', {})
        )
        
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        
        # 损失函数（分类器用）
        self.criterion = nn.BCELoss()
        self.backbone = self.build_backbone(config)  # 分类器
        self.loss_func = self.build_loss(config)
        
        # 损失权重参数
        self.lambda_cls = config.get('lambda_cls', 1.0)
        self.lambda_d = config.get('lambda_d', 1.0)
        self.lambda_g = config.get('lambda_g', 1.0)
    
    def generate_perturbation(self, x: torch.Tensor) -> Dict:
        """生成扰动 G(x) — 已包含 noise"""
        return self.generator(x)
    
    def discriminate(self, x: torch.Tensor, condition=None) -> torch.Tensor:
        """判别信号真伪 D(x, condition)"""
        return self.discriminator(x, condition)
    
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
        
        # 1. 提取真实样本（label==0）
        real_mask = labels == 0
        real_images = x[real_mask]
        real_labels = labels[real_mask]

        fake_mask = labels == 1
        fake_images = x[fake_mask]
        fake_labels = labels[fake_mask]

        if inference:
            with torch.no_grad():
                generator_output = self.generate_perturbation(fake_images)
                perturbation = generator_output['perturbation']
                perturbation= torch.cat([real_images,perturbation],dim=0)
                features = self.features({"image": perturbation})
                pred = self.classifier(features)
                prob = torch.softmax(pred, dim=1)[:, 1]
                scores = torch.softmax(pred, dim=1)
            return  {
                "pred_dict": 
                    {
                        "cls": pred,
                        "prob": prob,
                        "feat": features,
                        "scores": scores,
                    },
                "perturbation": perturbation,
            }
        
        
        # 2. 生成扰动（生成器现在返回 noise）
        condition=fake_images.detach()
        generator_output = self.generate_perturbation(fake_images)
        perturbation = generator_output['perturbation']
        
        # consistency_loss = generator_output.get('consistency_loss', None)
        
        perturbed_images = perturbation
        
        # 3. 分类器前向传播（只在需要时计算）
        if for_classifier:
            perturbed_images= torch.cat([real_images,perturbed_images],dim=0)
            features = self.features({"image": perturbed_images})
            pred = self.classifier(features)
            prob = torch.softmax(pred, dim=1)[:, 1]
            scores = torch.softmax(pred, dim=1)
        else:
            features, pred, prob, scores = None, None, None, None
        
        # 4. 判别器前向传播（使用 condition = noise）
        real_outputs = None
        fake_outputs = None
        real_feats =None
        fake_outputsd = None
        fake_feats = None
        fake_featsd = None  
       
        
        if for_discriminator or for_generator:
            # # 真实样本条件：使用全零向量（你可以替换为其它策略）
            if real_images.numel() > 0:
              
                real_outputs,real_feats = self.discriminate(real_images)
            
            # 生成样本判别（用于 generator training / overall evaluation）
            fake_outputs,fake_feats = self.discriminate(perturbed_images, condition)
            
            # detach 版本用于训练 discriminator（避免梯度流回 G）
            if for_discriminator:
                fake_outputsd ,fake_featsd= self.discriminate(perturbed_images.detach(), condition)
        
        result = {
            "real_data": {
                "image": real_images,
                "label": real_labels,
                "outputs": (real_outputs,real_feats)
            },
            "freq_data_dict": {
                "image": perturbed_images,
                "label": fake_labels,
                "outputs": (fake_outputs,fake_feats),
                "outputsd": (fake_outputsd,fake_featsd)
            },
            "pred_dict": {
                "cls": pred,
                "prob": prob,
                "feat": features,
                "scores": scores
            }
        }
        
        
            
        return result
    
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
        self.video_names = []
        return metric_batch_dict


