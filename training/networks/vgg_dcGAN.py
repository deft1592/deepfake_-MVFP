import torch
import torch.nn as nn
import torch.nn.functional as F

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

        # perturbation scale
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, freq):
        """
        freq: (B, 2C, H, W)
        """
        B, _, H, W = freq.shape

        # stochastic noise
        z = torch.randn(B, self.latent_dim, device=freq.device)
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

        perm = torch.randperm(B)
        lam = torch.rand(B, 1, 1, 1, device=delta.device)

        delta = lam * delta + (1 - lam) * delta[perm]
        return delta
        return self.alpha * delta




# =================================================
# DCGAN Discriminator (hinge loss)
# =================================================
class Discriminator(nn.Module):
    """
    DCGAN-style discriminator
    (logits output, no sigmoid)
    """
    def __init__(self, channels=3):
        super().__init__()

        self.net = nn.Sequential(
            # 256 → 128
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            # 128 → 64
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 64 → 32
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 32 → 16
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1)   # logits
        )

    def forward(self, x):
        return self.net(x)
