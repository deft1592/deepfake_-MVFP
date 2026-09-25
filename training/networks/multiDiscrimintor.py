import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VGGBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# =================================================
# VGG-style Generator (Frequency Domain)
# =================================================
class Generator(nn.Module):
    """
    VGG-style frequency perturbation generator
    """
    def __init__(self, image_size=256, latent_dim=16, channels=3):
        super().__init__()
        self.in_channels = channels * 2
        self.latent_dim = latent_dim

        # global noise
        self.noise_proj = nn.Linear(latent_dim, image_size * image_size)

        # ---------------- Encoder ----------------
        self.enc1 = VGGBlock(self.in_channels + 1, 64)
        self.down1 = nn.Conv2d(64, 128, 4, 2, 1)

        self.enc2 = VGGBlock(128, 128)
        self.down2 = nn.Conv2d(128, 256, 4, 2, 1)

        self.enc3 = VGGBlock(256, 256)

        # ---------------- Decoder ----------------
        self.up1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec1 = VGGBlock(128, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec2 = VGGBlock(64, 64)

        self.out = nn.Sequential(
            nn.Conv2d(64, self.in_channels, 3, 1, 1),
            nn.Tanh()
        )

        # perturbation scale (clamped in forward to avoid collapse / explosion)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, freq):
        """
        freq: (B, 2C, H, W)
        """
        B, _, H, W = freq.shape

        if self.training:
            z = torch.randn(B, self.latent_dim, device=freq.device)
        else:
            z = torch.zeros(B, self.latent_dim, device=freq.device)
        z_map = self.noise_proj(z).view(B, 1, H, W)

        x = torch.cat([freq, z_map], dim=1)

        x = self.enc1(x)
        x = F.leaky_relu(self.down1(x), 0.2, inplace=True)

        x = self.enc2(x)
        x = F.leaky_relu(self.down2(x), 0.2, inplace=True)

        x = self.enc3(x)

        x = F.leaky_relu(self.up1(x), 0.2, inplace=True)
        x = self.dec1(x)

        x = F.leaky_relu(self.up2(x), 0.2, inplace=True)
        x = self.dec2(x)

        delta = self.out(x)
        # Keep alpha in a stable range so G cannot collapse to zero scale
        alpha = self.alpha.clamp(0.02, 0.3)
        return alpha * delta

class DCGANBackbone(nn.Module):
    """
    Shared DCGAN-style discriminator backbone
    (logits output, no sigmoid)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)

class MultiFreqDiscriminator(nn.Module):
    """
    Multi-view frequency discriminator:
    - Magnitude
    - Phase
    - Radial spectrum
    - Band energy
    """
    BRANCH_NAMES = ("mag", "phase", "radial", "band")

    def __init__(self, channels=3, num_bands=4, enabled_branches=None):
        super().__init__()

        if enabled_branches is None:
            enabled_branches = self.BRANCH_NAMES
        enabled_branches = tuple(enabled_branches)
        invalid_branches = set(enabled_branches) - set(self.BRANCH_NAMES)
        if invalid_branches:
            raise ValueError(
                f"Unknown discriminator branches: {sorted(invalid_branches)}. "
                f"Expected a subset of {self.BRANCH_NAMES}."
            )
        self.enabled_branches = enabled_branches
        self.num_bands = num_bands
        # Keep a parameter so the existing optimizer construction also works
        # for the empty-subset ("none") ablation.
        self.disabled_anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        if "mag" in self.enabled_branches:
            self.D_mag = DCGANBackbone(1)
        if "phase" in self.enabled_branches:
            self.D_phase = DCGANBackbone(2)
        if "radial" in self.enabled_branches:
            self.D_radial = DCGANBackbone(1)
        if "band" in self.enabled_branches:
            self.D_band = DCGANBackbone(num_bands)
    
    def _fft(self, x):
        """
        x: (B, C, H, W)
        return: fft shifted complex tensor
        """
        gray = x.mean(dim=1)  # (B, H, W)
        fft = torch.fft.fft2(gray, norm="ortho")
        fft = torch.fft.fftshift(fft)
        return fft
    
    def _magnitude(self, x):
        fft = self._fft(x)
        mag = torch.log1p(torch.abs(fft))
        return mag.unsqueeze(1)  # (B,1,H,W)
    
    def _phase(self, x):
        fft = self._fft(x)
        mag = torch.abs(fft).clamp_min(1e-6)
        phase = torch.stack((fft.real / mag, fft.imag / mag), dim=1)
        return phase
    

    def _radial_spectrum(self, x):
        B, _, H, W = x.shape
        fft = self._fft(x)
        mag = torch.abs(fft)

        cy, cx = H // 2, W // 2
        y, x_grid = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing="ij"
        )
        r = torch.sqrt((x_grid - cx) ** 2 + (y - cy) ** 2)
        r = r / r.max()

        radial = torch.zeros_like(mag)
        for i in range(B):
            radial[i] = mag[i] * r

        radial = torch.log1p(radial)
        return radial.unsqueeze(1)


    def _band_energy(self, x):
        fft = self._fft(x)
        mag = torch.abs(fft)

        B, H, W = mag.shape
        cy, cx = H // 2, W // 2

        y, x_grid = torch.meshgrid(
            torch.arange(H, device=x.device),
            torch.arange(W, device=x.device),
            indexing="ij"
        )
        r = torch.sqrt((x_grid - cx) ** 2 + (y - cy) ** 2)
        r = r / r.max()

        bands = []
        edges = torch.linspace(0, 1, self.num_bands + 1, device=x.device)

        for i in range(self.num_bands):
            mask = (r >= edges[i]) & (r < edges[i + 1])
            band = mag * mask
            bands.append(torch.log1p(band))

        band_energy = torch.stack(bands, dim=1)  # (B, num_bands, H, W)
        return band_energy

    def forward(self, x):
        outputs = {}
        if "mag" in self.enabled_branches:
            outputs["mag"] = self.D_mag(self._magnitude(x))
        if "phase" in self.enabled_branches:
            outputs["phase"] = self.D_phase(self._phase(x))
        if "radial" in self.enabled_branches:
            outputs["radial"] = self.D_radial(self._radial_spectrum(x))
        if "band" in self.enabled_branches:
            outputs["band"] = self.D_band(self._band_energy(x))
        return outputs
