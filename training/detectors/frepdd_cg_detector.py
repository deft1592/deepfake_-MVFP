

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
from training.networks.multiDiscrimintor  import Generator as FrequencyPerturbationGenerator

from training.networks.multiDiscrimintor  import MultiFreqDiscriminator as SpectralDiscriminator

# from training.networks.vgg_dcGAN import Generator as FrequencyPerturbationGenerator

# from training.networks.vgg_dcGAN  import Discriminator as SpectralDiscriminator



device = "cuda" if torch.cuda.is_available() else "cpu"
logger = logging.getLogger(__name__)



class LowFreqPerturbationGenerator(nn.Module):
    """
    x -> FFT -> Frequency Generator -> IFFT -> spatial perturbation
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        image_size = config.get('resolution', 256)
        channels = config.get('img_channels', 3)
        latent_dim = config.get('latent_dim', 32)
        

        self.freq_generator = FrequencyPerturbationGenerator(
            image_size=image_size,
            latent_dim=latent_dim,
            channels=channels
        )

    def forward(self, x: torch.Tensor) -> Dict:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # =========================
        # FFT
        # =========================
        x_fft = torch.fft.fft2(x, norm='ortho')
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))

        # real & imag split
        freq_input = torch.cat(
            [x_fft_shift.real, x_fft_shift.imag], dim=1
        )  # (B, 2C, H, W)

        # =========================
        # Frequency perturbation
        # =========================
        delta_freq = self.freq_generator(freq_input)

        # split back
        delta_real, delta_imag = torch.chunk(delta_freq, 2, dim=1)
        delta_fft = torch.complex(delta_real, delta_imag)

        # =========================
        # IFFT
        # =========================
        perturbed_fft = delta_fft
        perturbed_fft = perturbed_fft.to(torch.complex64)

        perturbed_fft = torch.fft.ifftshift(perturbed_fft, dim=(-2, -1))

        perturbation = torch.fft.ifft2(
            perturbed_fft, norm='ortho'
        ).real

        # =========================
        # Loss: only punish over-budget perturbation (avoid collapse-to-zero)
        # =========================
        pert_rms = torch.sqrt(torch.mean(perturbation ** 2) + 1e-8)
        budget = float(self.config.get("pert_budget", 0.02))
        compression_loss = F.relu(pert_rms - budget) ** 2

        return {
            "perturbation": perturbation,
            "compression_loss": compression_loss,
            "pert_rms": pert_rms.detach(),
        }


    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)
    
    def init_weights(self, pretrained_path: Union[bool, str] = None):
        if pretrained_path is None:
            self._initialize_weights()
        elif isinstance(pretrained_path, str):
            self.load_state_dict(torch.load(pretrained_path))


class SpectralDiscriminatorWrapper(nn.Module):
    """
    基于SpectralDiscriminator的判别器包装器
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 获取配置参数
        img_channels = config.get('img_channels', 3)
        img_size = config.get('resolution', 256)
        enabled_branches = config.get("enabled_branches")
        num_bands = config.get("num_bands", 4)
        
        # SpectralDiscriminator 判别器结构
        self.discriminator = SpectralDiscriminator(
            img_channels,
            num_bands=num_bands,
            enabled_branches=enabled_branches,
        )
        self.discriminator = self.discriminator.to(device)

    @property
    def enabled_branches(self):
        return self.discriminator.enabled_branches
    
    def forward(self, x: torch.tensor) -> torch.Tensor:
        """判别图像真伪"""
        return self.discriminator(x)
    
        
    def _initialize_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)
    
    def init_weights(self, pretrained_path: Union[bool, str] = None):
        """权重初始化"""
        if pretrained_path is None:
            # 默认初始化
            self._initialize_weights()
        elif isinstance(pretrained_path, str):
            # 加载预训练权重
            self.load_state_dict(torch.load(pretrained_path))


@DETECTOR.register_module(module_name="frepddCg")
class FrepddCgDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.prob, self.label = [], []
        self.video_names = []
        self.correct, self.total = 0, 0
        
        # 创建生成器和判别器
        gen_cfg = dict(config.get("generator", {}) or {})
        gen_cfg.setdefault("resolution", config.get("resolution", 256))
        gen_cfg.setdefault("img_channels", config.get("img_channels", 3))
        gen_cfg.setdefault("latent_dim", config.get("latent_dim", 32))
        gen_cfg.setdefault("pert_budget", config.get("pert_budget", 0.02))
        self.generator = LowFreqPerturbationGenerator(gen_cfg)
        
        self.discriminator = SpectralDiscriminatorWrapper(
            {
                "img_channels": config.get("img_channels", 3),
                "resolution": config.get("resolution", 256),
                **(config.get("discriminator", {}) or {}),
            }
        )
        
        self.backbone = self.build_backbone(config)  # 分类器
        self.loss_func = self.build_loss(config)

        # self.register_buffer("lam_mag",    torch.tensor(1.0))
        # self.register_buffer("lam_radial", torch.tensor(0.5))
        # self.register_buffer("lam_band",   torch.tensor(0.3))
        # self.register_buffer("lam_phase",  torch.tensor(0.2))


        default_branch_weights = {
            "mag": 1.0,
            "radial": 0.5,
            "band": 0.3,
            "phase": 0.2,
        }
        configured_weights = (
            (config.get("discriminator", {}) or {}).get("branch_weights", {})
        )
        self.branch_weights = {
            name: float(configured_weights.get(name, default_weight))
            for name, default_weight in default_branch_weights.items()
        }
        self.enabled_discriminator_branches = self.discriminator.enabled_branches
        logger.info(
            "Enabled discriminator branches: %s",
            ", ".join(self.enabled_discriminator_branches),
        )
    

    def band_dropout_mask(self,H, W, drop_width=8):
        """
        Drop a random radial frequency band
        """
        center_y, center_x = H // 2, W // 2
        Y, X = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="ij"
        )
        dist = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)

        r_max = dist.max()
        r0 = torch.rand(1).item() * (r_max - drop_width)
        r1 = r0 + drop_width

        mask = ~((dist >= r0) & (dist <= r1))
        return mask.float()

    def apply_band_dropout(self,x, drop_width=8):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))

        mask = self.band_dropout_mask(H, W).to(x.device)
        mask = mask[None, None, :, :]

        fft = fft * mask

        x_drop = torch.fft.ifft2(
            torch.fft.ifftshift(fft),
            norm='ortho'
        ).real

        return x_drop

    def all_d_hinge_loss(self, real_logits, fake_logits):
        if not self.enabled_discriminator_branches:
            return next(self.parameters()).new_zeros(())
        return sum(
            self.branch_weights[name]
            * self.d_hinge_loss(real_logits[name], fake_logits[name])
            for name in self.enabled_discriminator_branches
        )

    def d_hinge_loss(self, real_logits, fake_logits):
        """
        Hinge loss for discriminator
        """
        loss_real = torch.mean(F.relu(1.0 - real_logits))
        loss_fake = torch.mean(F.relu(1.0 + fake_logits))
        return loss_real + loss_fake



    def g_hinge_loss(self, outputs):
        if not self.enabled_discriminator_branches:
            return next(self.parameters()).new_zeros(())
        return sum(
            -self.branch_weights[name] * outputs[name].mean()
            for name in self.enabled_discriminator_branches
        )

    # def g_hinge_loss(self, outputs):
    #     return -outputs.mean()
        

    def generate_perturbation(self, x: torch.Tensor) -> Dict:
        """生成扰动 G(x)"""
        scale = float(self.config.get("perturbation_scale", 1.0))
        if scale == 0.0:
            zero = x.new_zeros(())
            return {
                "perturbation": torch.zeros_like(x),
                "pert_rms": zero.detach(),
                "compression_loss": zero,
            }

        output = self.generator(x)
        if scale == 1.0:
            return output

        output = dict(output)
        perturbation = output["perturbation"] * scale
        pert_rms = torch.sqrt(torch.mean(perturbation ** 2) + 1e-8)
        budget = float(self.config.get("pert_budget", 0.02))
        output["perturbation"] = perturbation
        output["pert_rms"] = pert_rms.detach()
        output["compression_loss"] = F.relu(pert_rms - budget) ** 2
        return output
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        # """判别信号真伪 D(x)"""
        drop_prob = float(self.config.get("band_dropout_prob", 0.2))
        if self.training and drop_prob > 0 and torch.rand(1, device=x.device) < drop_prob:
            x = self.apply_band_dropout(x, drop_width=8)
        return self.discriminator(x)
    
    def build_backbone(self, config):
        backbone_class = BACKBONE[config["backbone_name"]]
        model_config = dict(config["backbone_config"])
        model_config["pretrained"] = config.get("pretrained")
        backbone = backbone_class(model_config)

        # Modern backbone adapters load and validate their own pretrained weights.
        if getattr(backbone, "loads_own_pretrained", False):
            return backbone

        # if donot load the pretrained weights, fail to get good results
        state_dict = torch.load(config["pretrained"], map_location="cpu")
        for name, weights in state_dict.items():
            if "pointwise" in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k: v for k, v in state_dict.items() if "fc" not in k}
        
        backbone.load_state_dict(state_dict, False)
        logger.info("Load pretrained model successfully!")
        return backbone

    def build_loss(self, config):
        # prepare the loss function
        loss_class = LOSSFUNC[config["loss_func"]]
        loss_func = loss_class()
        
        return loss_func

    def features(self, data_dict: dict) -> torch.tensor:
        drop_prob = float(self.config.get("band_dropout_prob", 0.2))
        if self.training and drop_prob > 0 and torch.rand(1, device=data_dict["image"].device) < drop_prob:
            data_dict["image"] = self.apply_band_dropout(data_dict["image"], drop_width=8)
        return self.backbone.features(data_dict["image"])

    def classifier(self, features: torch.tensor) -> torch.tensor:
        return self.backbone.classifier(features)

    def get_losses(self, data_dict: dict, pred: dict) -> dict:
        cls_loss = self.get_cls_loss(data_dict, pred)
        d_loss = self.get_d_loss(data_dict, pred)
        g_loss = self.get_g_loss(data_dict, pred)
        return {
            "overall": cls_loss + d_loss + g_loss,
            "cls_loss": cls_loss,
            "d_loss": d_loss,
            "g_loss": g_loss
        }

    def get_d_loss(self, data_dict: dict, pred: dict) -> torch.Tensor:
        if not self.enabled_discriminator_branches:
            return data_dict["image"].new_zeros(())
        real_data = pred["real_data"]
        freq_data_dict = pred['freq_data_dict']

        if real_data["outputs"] is None:
            # Keep every discriminator branch in the autograd graph. Different
            # DDP ranks can receive different label mixtures; a detached zero on
            # one rank would skip its gradient collective and hang the others.
            fake_logits = freq_data_dict["outputsd"]
            return sum(logits.sum() for logits in fake_logits.values()) * 0.0

        real_logits = real_data["outputs"]
        fake_logits = freq_data_dict["outputsd"]

        #return self.d_hinge_loss(real_logits, fake_logits)
        return self.all_d_hinge_loss(real_logits, fake_logits)

    def get_g_loss(self, data_dict: dict, pred: dict) -> torch.Tensor:
        
        adv_weight = self.config.get("g_adv_weight", 0.5)
        compression_weight = self.config.get("g_compression_weight", 0.5)
        task_weight = self.config.get("g_task_weight", 0.2)
        freq_data_dict = pred['freq_data_dict']

        fake_logits = freq_data_dict["outputs"]
        if self.enabled_discriminator_branches:
            g_loss = adv_weight * self.g_hinge_loss(fake_logits)
        else:
            g_loss = data_dict["image"].new_zeros(())

        if "compression_loss" in freq_data_dict:
            compression_loss = freq_data_dict["compression_loss"]
            if self.config.get("normalize_compression_loss", False):
                budget = max(float(self.config.get("pert_budget", 0.02)), 1e-6)
                compression_loss = compression_loss / (budget ** 2)
            g_loss = g_loss + compression_weight * compression_loss

        pred_dict = pred.get("pred_dict", {})
        if task_weight > 0 and pred_dict.get("cls", None) is not None:
            g_loss = g_loss + task_weight * self.loss_func(pred_dict["cls"], data_dict["label"])

        return g_loss

    

    def get_cls_loss(self, data_dict: dict, pred: dict) -> torch.Tensor:
        """单独计算分类器损失"""
        pred_dict = pred['pred_dict']
        if pred_dict["cls"] is not None:
            label = data_dict["label"]
            cls_pred = pred_dict["cls"]
            label_smoothing = float(self.config.get("label_smoothing", 0.0))
            class_weights = self.config.get("class_weights")
            weight = None
            if class_weights is not None:
                weight = cls_pred.new_tensor(class_weights)
            if label_smoothing > 0:
                cls_loss = F.cross_entropy(
                    cls_pred,
                    label,
                    weight=weight,
                    label_smoothing=label_smoothing,
                )
            elif weight is not None:
                cls_loss = F.cross_entropy(cls_pred, label, weight=weight)
            else:
                cls_loss = self.loss_func(cls_pred, label)
            
            
            return cls_loss
        
        return torch.tensor(0.0, device=data_dict["label"].device)
    
    
    def forward(self, data_dict: dict, inference=False,for_generator= False,for_discriminator=False,for_classifier =False) -> dict:
        x = data_dict["image"]
        labels = data_dict["label"]

        # 1. 提取真实样本（label=0）
        real_mask = labels == 0
        real_images = x[real_mask]
        real_labels = labels[real_mask]
        
        # 1. 提取fake样本（label=1）
        fake_images = x[labels==1]
        fake_labels = labels[labels==1]

        if inference:
            # 推理模式下不计算梯度
            with torch.no_grad():
                
        
                # real_perturbation = real_generator_output['perturbation']
                # fake_generator_output = self.generate_perturbation(fake_images)
                # fake_perturbation = fake_generator_output['perturbation']
                # perturbation = torch.cat([real_perturbation, fake_perturbation], dim=0)
                
                generator_output = self.generate_perturbation(x)
                perturbation = generator_output['perturbation']

                perturbed_images = x + perturbation
                features = self.features({"image": perturbed_images})
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
                "perturbed_image": perturbed_images 
            }
        
        
        # real_generator_output = self.generate_perturbation(real_images)
        
        # real_perturbation = real_generator_output['perturbation']
        # fake_generator_output = self.generate_perturbation(fake_images)
        # fake_perturbation = fake_generator_output['perturbation']
        # perturbation = torch.cat([real_perturbation, fake_perturbation], dim=0)

        # compression_loss = real_generator_output.get('compression_loss', None)+fake_generator_output.get('compression_loss', None)


        # 2. 生成扰动
        generator_output = self.generate_perturbation(x)
        perturbation = generator_output['perturbation']
        
        compression_loss = generator_output.get('compression_loss', None)
        

        
        perturbed_images = x + perturbation 
        
        # 4. 分类器前向传播（只在需要时计算）
        if  for_classifier or for_generator:
            features = self.features({"image": perturbed_images})
            pred = self.classifier(features)
            prob = torch.softmax(pred, dim=1)[:, 1]
            scores = torch.softmax(pred, dim=1)
        else:
            features, pred, prob, scores = None, None, None, None
        
        
        # 5. 判别器前向传播（只在需要时计算）
        if for_discriminator:
            # 真实样本判别
            real_outputs = self.discriminate(real_images) if real_images.numel() > 0 else None
            
            # 生成样本判别
            fake_outputs = None
            fake_outputsd = self.discriminate(perturbed_images.detach())
        elif for_generator:
            real_outputs = None
            fake_outputs = self.discriminate(perturbed_images)
            fake_outputsd = None
        else:
            real_outputs = None
            fake_outputs = None
            fake_outputsd = None
        
        
        result = {
            "real_data": {
                "image": real_images,
                "label": real_labels,
                "outputs": real_outputs
            },
            "freq_data_dict": {
                "image": perturbation,
                "label": labels,
                "outputs": fake_outputs,
                "outputsd": fake_outputsd
            },
            "pred_dict": {
                "cls": pred,
                "prob": prob,
                "feat": features,
                "scores": scores
            }
        }
        
        # 添加一致性损失到结果中
        if compression_loss is not None:
            result["freq_data_dict"]["compression_loss"] = compression_loss
        if "pert_rms" in generator_output:
            result["freq_data_dict"]["pert_rms"] = generator_output["pert_rms"]
            
        return result
            
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict["label"]
        pred = pred_dict["cls"]
        # compute metrics for batch data
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        metric_batch_dict = {"acc": acc, "auc": auc, "eer": eer, "ap": ap}
        # we dont compute the video-level metrics for training
        self.video_names = []
        return metric_batch_dict
