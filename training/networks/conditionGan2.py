import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2
import torch.nn.utils as utils

class ImprovedLowFreqConditionalGenerator(nn.Module):
    """改进的轻量级条件生成器：加入 skip-connection，防止生成器重建整图"""
    def __init__(self, image_size=256, latent_dim=64, channels=3):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.channels = channels
        
        # ------------ 你的编码器保持不变 ------------
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.encoded_size = image_size // 8
        self.encoded_dim = 256 * (self.encoded_size ** 2)

        # ------------ 你的 fusion 保持不变 ------------
        self.fusion = nn.Sequential(
            nn.Linear(self.encoded_dim, 1024),
            nn.InstanceNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(512, 256 * self.encoded_size * self.encoded_size),
        )

        # ------------ decoder 保持不变 ------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(128),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            ResidualBlock(64),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )


    def forward(self, x):
        encoded = self.encoder(x)
        B = encoded.size(0)

        encoded_flat = encoded.view(B, -1)
        fused = self.fusion(encoded_flat)

        fused_reshaped = fused.view(B, 256, self.encoded_size, self.encoded_size)
        out = self.decoder(fused_reshaped)   # (B,3,H,W)

        # ⭐ Skip 通道（最关键的一行）
        out = out + x

        # 限制输出范围（保持与你原来 Tanh 的风格一致）
        out = torch.clamp(out, -1, 1)

        return out


class ResidualBlock(nn.Module):
    """残差块：增强梯度流动和表达能力"""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
        )
        
    def forward(self, x):
        return x + self.block(x)



class ImprovedConditionalSpectralDiscriminator(nn.Module):
    """
    Unconditional spectral discriminator:
    - No condition input
    - Uses spatial + frequency branches
    - Returns (logit, feat)
    """
    def __init__(self, channels=3, image_size=256):
        super().__init__()
        self.image_size = image_size

        # spatial path: now input is only image channels
        self.spatial_path = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, 32, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.2),
        )

        # Frequency branch  (mag + phase)
        self.freq_path = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(2, 32, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.utils.spectral_norm(nn.Conv2d(32, 64, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.2),
        )

        # fusion
        self.class_conv = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128 + 64, 256, 4, 2, 1, bias=False)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.utils.spectral_norm(nn.Linear(256, 1))

    # ============================
    # frequency feature extractor
    # ============================
    def get_frequency_features(self, x):
        # x: (B, C, H, W)
        gray = x.mean(dim=1)  # (B, H, W)

        fft = torch.fft.fft2(gray)
        fft = torch.fft.fftshift(fft)

        mag = torch.log1p(torch.abs(fft))   # magnitude
        phase = torch.angle(fft)            # phase

        freq = torch.stack([mag, phase], dim=1)  # (B,2,H,W)
        return freq

    # ============================
    # forward
    # ============================
    def forward(self, x):
        """
        x: (B,3,H,W)
        """
        b = x.shape[0]

        # spatial branch
        spatial_features = self.spatial_path(x)

        # freq branch
        freq_features = self.get_frequency_features(x)
        freq_spatial_features = self.freq_path(freq_features)

        # match shapes
        if freq_spatial_features.shape[2:] != spatial_features.shape[2:]:
            freq_spatial_features = F.interpolate(
                freq_spatial_features,
                size=spatial_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        # fuse
        combined = torch.cat([spatial_features, freq_spatial_features], dim=1)

        x_out = self.class_conv(combined)
        feat = self.pool(x_out).view(b, -1)
        logit = self.fc(feat).view(b, 1)

        return logit, feat
