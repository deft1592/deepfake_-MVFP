import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2


import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Noise Injection (StyleGAN-style)
# -------------------------------------------------
class NoiseInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        noise = torch.randn(
            x.size(0), 1, x.size(2), x.size(3),
            device=x.device, dtype=x.dtype
        )
        return x + self.weight * noise


# -------------------------------------------------
# Residual Block with Noise
# -------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.noise1 = NoiseInjection(channels)

        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(channels)
        self.noise2 = NoiseInjection(channels)

    def forward(self, x):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.noise1(h)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.noise2(h)

        return x + h


# -------------------------------------------------
# Generator
# -------------------------------------------------
class Generator(nn.Module):
    """
    Stochastic frequency-domain perturbation generator
    (strong diversity, stable training)
    """
    def __init__(self, image_size=256, latent_dim=16, channels=3):
        super().__init__()
        self.in_channels = channels * 2
        self.latent_dim = latent_dim

        # global noise projection
        self.noise_proj = nn.Linear(latent_dim, image_size * image_size)

        # ---------------- Encoder ----------------
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channels + 1, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # ---------------- Bottleneck ----------------
        self.middle = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
        )

        # ---------------- Decoder ----------------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, self.in_channels, 3, 1, 1),
            nn.Tanh()
        )

        # energy control
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, freq):
        B, _, H, W = freq.shape

        # stochastic latent
        z = torch.randn(B, self.latent_dim, device=freq.device)
        z_map = self.noise_proj(z).view(B, 1, H, W)

        x = torch.cat([freq, z_map], dim=1)

        feat = self.encoder(x)
        feat = self.middle(feat)
        delta = self.decoder(feat)
        perm = torch.randperm(B)
        lam = torch.rand(B, 1, 1, 1, device=delta.device)

        return lam * delta + (1 - lam) * delta[perm]

class Discriminator(nn.Module):
    """
    Weak frequency-aware discriminator
    (hinge loss, anti-collapse)
    """
    def __init__(self, channels=3):
        super().__init__()

        # -------- spatial path --------
        self.spatial = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # -------- frequency path (magnitude only) --------
        self.freq = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # -------- classifier --------
        self.classifier = nn.Sequential(
            nn.Conv2d(128 + 64, 192, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 1)   # logits
        )

    @torch.no_grad()
    def _fft_mag(self, x):
        gray = x.mean(dim=1)
        fft = torch.fft.fft2(gray, norm='ortho')
        fft = torch.fft.fftshift(fft)
        mag = torch.log1p(torch.abs(fft))
        return mag.unsqueeze(1)

    def forward(self, x):
        s_feat = self.spatial(x)

        f_mag = self._fft_mag(x)
        f_feat = self.freq(f_mag)

        if f_feat.shape[-2:] != s_feat.shape[-2:]:
            f_feat = F.interpolate(
                f_feat,
                size=s_feat.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        feat = torch.cat([s_feat, f_feat], dim=1)
        return self.classifier(feat)
