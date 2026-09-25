import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


device = "cuda" if torch.cuda.is_available() else "cpu"
class DiffusionGenerator(nn.Module):
    """
    完整的扩散模型生成器，包含反向去噪过程
    输入输出尺寸保持一致
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # 获取配置参数
        self.img_channels = config.get('img_channels', 3)
        self.img_size = config.get('resolution', 256)
        self.ngf = config.get('ngf', 32)  # 减少通道数以节省显存
        self.latent_dim = config.get('latent_dim', 64)  # 减少潜在维度
        self.timesteps = config.get('timesteps', 500)  # 减少时间步数以节省显存
        
        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.ngf * 4),
            nn.ReLU(),
            nn.Linear(self.ngf * 4, self.ngf * 4)
        )
        
        # U-Net 主干网络
        self.unet = UNet(self.img_channels, self.ngf, self.latent_dim,self.ngf*4)
        
        # # 压缩损失网络（预训练VGG）
        # self.compression_net = self.build_compression_net()
        
        # 扩散参数
        self.beta_schedule = self.create_beta_schedule(self.timesteps)
        self.alpha = 1.0 - self.beta_schedule
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.alpha_bar_prev = F.pad(self.alpha_bar[:-1], (1, 0), value=1.0)
        
        # 计算反向扩散参数
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recip_minus_alpha_bar = torch.sqrt(1.0 / self.alpha_bar - 1)
        
        # 后验方差
        self.posterior_variance = (
            self.beta_schedule * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        
        # 注册缓冲区（使用不同名称）
        self.register_buffer('beta_schedule_buffer', self.beta_schedule)
        self.register_buffer('alpha_buffer', self.alpha)
        self.register_buffer('alpha_bar_buffer', self.alpha_bar)
        self.register_buffer('alpha_bar_prev_buffer', self.alpha_bar_prev)
        self.register_buffer('sqrt_alpha_bar_buffer', self.sqrt_alpha_bar)
        self.register_buffer('sqrt_one_minus_alpha_bar_buffer', self.sqrt_one_minus_alpha_bar)
        self.register_buffer('sqrt_recip_alpha_bar_buffer', self.sqrt_recip_alpha_bar)
        self.register_buffer('sqrt_recip_minus_alpha_bar_buffer', self.sqrt_recip_minus_alpha_bar)
        self.register_buffer('posterior_variance_buffer', self.posterior_variance)
    
    def to(self, device):
        """重写to方法，确保所有缓冲区移动到正确设备"""
        super().to(device)
        # 确保所有缓冲区也在设备上
        for buffer in self.buffers():
            buffer.data = buffer.data.to(device)
        return self
    
    # def build_compression_net(self):
    #     """构建压缩损失网络（使用预训练VGG）"""
    #     vgg = vgg16(pretrained=True).features[:16]
    #     # 冻结参数
    #     for param in vgg.parameters():
    #         param.requires_grad = False
    #     return vgg
    
    def create_beta_schedule(self, timesteps, schedule='cosine'):
        """创建噪声调度"""
        if schedule == 'linear':
            beta = torch.linspace(0.0001, 0.02, timesteps)
        elif schedule == 'cosine':
            steps = timesteps + 1
            s = 0.008
            x = torch.linspace(0, timesteps, steps)
            alpha_bar = torch.cos((x / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        return beta
    
    def forward_diffusion(self, x0, t):
        """前向扩散过程：添加噪声"""
        # 确保所有张量在同一设备上
        device = x0.device
        
        # 将缓冲区移动到当前设备
        sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        
        # 获取当前时间步的值
        sqrt_alpha_bar_t = sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        
        # 计算噪声
        noise = torch.randn_like(x0)
        
        # 添加噪声
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        
        return x_t, noise
    
    def reverse_diffusion(self, x_t, t, cond=None):
        """反向扩散过程：预测噪声"""
        # 获取时间嵌入
        t_emb = self.time_embed(t.float().view(-1, 1))
        
        # 预测噪声
        pred_noise = self.unet(x_t, t_emb, cond)
        
        return pred_noise
    
    def predict_x0_from_noise(self, x_t, t, noise):
        """从噪声预测原始图像"""
        
        
        sqrt_recip_alpha_bar = self.sqrt_recip_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_recip_minus_alpha_bar = self.sqrt_recip_minus_alpha_bar[t].view(-1, 1, 1, 1)
        
        sqrt_recip_alpha_bar=sqrt_recip_alpha_bar.to(x_t.device)
        sqrt_recip_minus_alpha_bar=sqrt_recip_minus_alpha_bar.to(x_t.device)

        return sqrt_recip_alpha_bar * x_t - sqrt_recip_minus_alpha_bar * noise
    
    def q_posterior_mean_variance(self, x0, x_t, t):
        """计算后验分布的均值和方差"""
        # 获取批次大小和设备
        batch_size = x0.size(0)
        device = x0.device
        
        # 获取当前时间步的值
        alpha_bar_t = self.alpha_bar[t].view(batch_size, 1, 1, 1).to(device)
        alpha_bar_prev_t = self.alpha_bar_prev[t].view(batch_size, 1, 1, 1).to(device)
        
        # 计算系数
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        # 计算后验均值
        posterior_mean = (
            (sqrt_alpha_bar_t * x0 - sqrt_one_minus_alpha_bar_t * x_t) / 
            (alpha_bar_t - alpha_bar_prev_t)
        )
        
        # 后验方差
        posterior_variance = self.posterior_variance[t].view(batch_size, 1, 1, 1).to(device)
        
        return posterior_mean, posterior_variance
    
    def reconstruct_from_noise(self, x_t, t, pred_noise):
        """从噪声重建图像（使用反向去噪过程）"""
        # 预测原始图像
        pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)
        
        # 计算后验均值和方差
        model_mean, model_variance = self.q_posterior_mean_variance(pred_x0, x_t, t)
        
        # 如果不是最后一步，添加噪声
        if t[0] > 0:  # 假设所有样本在同一时间步
            noise = torch.randn_like(x_t)
            x_recon = model_mean + torch.sqrt(model_variance) * noise
        else:
            x_recon = model_mean
        
        return x_recon
    
    def compression_loss(self, x0, t, pred_noise):
        """计算压缩损失（感知损失）"""
        # 获取当前时间步的alpha_bar
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        
        
        alpha_bar_t=alpha_bar_t.to(x0.device)
        # 创建加噪图像
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * pred_noise
        
        # 重建图像
        x_recon = self.reconstruct_from_noise(x_t, t, pred_noise)

        pred=x_recon
        target=x0

        loss = F.mse_loss(pred, target)
        
        return loss
    
    def forward(self, x, cond=None):
        """训练时前向传播"""
        # 随机选择时间步
        t = torch.randint(0, self.timesteps, (x.size(0),), device=x.device)
        
        # 前向扩散
        x_t, noise = self.forward_diffusion(x, t)
        
        # 预测噪声
        pred_noise = self.reverse_diffusion(x_t, t, cond)
        
        # 这里使用标准扩散模型的损失
        comp_loss = F.mse_loss(pred_noise, noise)
        
        return {
            "pred_noise": pred_noise,
            "true_noise": noise,
            "comp_loss": comp_loss
        }
    
    def sample(self, model, cond=None, shape=None):
        """完整的反向去噪采样过程"""
        # 如果没有提供形状，使用条件输入的形状
        if shape is None and cond is not None:
            shape = cond.shape
        elif shape is None:
            shape = (1, self.img_channels, self.img_size, self.img_size)
        
        # 初始化随机噪声
        device = next(model.parameters()).device
        x_t = torch.randn(shape, device=device)
        
        # 逐步去噪
        for i in range(self.timesteps - 1, -1, -1):
            # 创建时间步张量
            t = torch.full((x_t.size(0),), i, device=device)
            
            # 计算均值和方差
            pred_noise = self.reverse_diffusion(x_t, t, cond)
            pred_x0 = self.predict_x0_from_noise(x_t, t, pred_noise)
            model_mean, model_variance = self.q_posterior_mean_variance(pred_x0, x_t, t)
            
            # 如果不是最后一步，添加噪声
            if i > 0:
                noise = torch.randn_like(x_t)
                x_t = model_mean + torch.sqrt(model_variance) * noise
            else:
                x_t = model_mean
        
        return x_t
    
    def generate(self, cond=None, shape=None):
        """生成扰动（使用完整的反向去噪）"""
        return self.sample(self.unet, cond, shape)
    
    def ddim_sample(self, cond=None, shape=None, steps=20):
        """使用DDIM加速采样"""
        # 如果没有提供形状，使用条件输入的形状
        if shape is None and cond is not None:
            shape = cond.shape
        elif shape is None:
            shape = (1, self.img_channels, self.img_size, self.img_size)
        
        # 初始化随机噪声
        device = next(self.unet.parameters()).device
        x_t = torch.randn(shape, device=device)
        
        # 创建DDIM时间表
        step_size = self.timesteps // steps
        timesteps = list(range(0, self.timesteps, step_size))[::-1]
        
        for i, t in enumerate(timesteps):
            # 创建时间步张量
            t_tensor = torch.full((x_t.size(0),), t, device=device)
            
            # 预测噪声
            pred_noise = self.reverse_diffusion(x_t, t_tensor, cond)
            
            # 计算alpha和beta
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            
            # 计算预测的原始图像
            pred_x0 = self.predict_x0_from_noise(x_t, t_tensor, pred_noise)
            
            # 如果是最后一步，直接返回
            if i == len(timesteps) - 1:
                x_t = pred_x0
                break
            
            # 计算下一个时间步
            next_t = timesteps[i+1]
            alpha_bar_next = self.alpha_bar[next_t]
            
            # 计算方向
            direction = torch.sqrt(1 - alpha_bar_next) * pred_noise
            
            # 更新x_t
            x_t = torch.sqrt(alpha_bar_next) * pred_x0 + direction
        
        return x_t

class UNet(nn.Module):
    """修正后的U-Net架构，确保尺寸匹配"""
    def __init__(self, in_channels, ngf, latent_dim, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.ngf = ngf
        self.latent_dim = latent_dim
        self.time_emb_dim = time_emb_dim
        
        # 下采样路径
        self.down1 = DownBlock(in_channels, ngf)
        self.down2 = DownBlock(ngf, ngf * 2)
        self.down3 = DownBlock(ngf * 2, ngf * 4)
        self.down4 = DownBlock(ngf * 4, ngf * 8)
        
        # 瓶颈层
        self.bottleneck = Bottleneck(ngf * 8, latent_dim, time_emb_dim)
        
        # 上采样路径
        self.up4 = UpBlock(latent_dim + ngf * 8, ngf * 8)
        self.up3 = UpBlock(ngf * 8 + ngf * 4, ngf * 4)
        self.up2 = UpBlock(ngf * 4 + ngf * 2, ngf * 2)
        self.up1 = UpBlock(ngf * 2 + ngf, ngf)
        
        # 输出层
        self.out = nn.Conv2d(ngf, in_channels, 3, 1, 1)
        
        # 时间嵌入处理
        self.time_emb_proj = nn.Sequential(
            nn.Linear(time_emb_dim, ngf * 4),
            nn.ReLU(),
            nn.Linear(ngf * 4, ngf * 4)
        )
    
    def forward(self, x, t_emb, cond=None):
        # 处理时间嵌入
        t_emb = self.time_emb_proj(t_emb)
        
        # 下采样
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        d3, skip3 = self.down3(d2)
        d4, skip4 = self.down4(d3)
        
        # 瓶颈层
        b = self.bottleneck(d4, t_emb)
        
        # 上采样
        u4 = self.up4(b, skip4)
        u3 = self.up3(u4, skip3)
        u2 = self.up2(u3, skip2)
        u1 = self.up1(u2, skip1)
        
        # 输出
        return self.out(u1)
    
    
class DownBlock(nn.Module):
    """下采样块，确保尺寸减半"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.pool = nn.AvgPool2d(2)  # 使用池化确保精确减半
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        # 第一层卷积
        x = self.act(self.norm(self.conv1(x)))
        
        # 保存跳跃连接
        skip = x
        
        # 第二层卷积
        x = self.act(self.norm(self.conv2(x)))
        
        # 下采样
        x = self.pool(x)
        
        return x, skip
    
    
class UpBlock(nn.Module):
    """上采样块，确保尺寸匹配"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x, skip):
        # 上采样到skip的尺寸
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        
        # 拼接跳跃连接
        x = torch.cat([x, skip], dim=1)
        
        # 第一层卷积
        x = self.act(self.norm(self.conv1(x)))
        
        # 第二层卷积
        x = self.act(self.norm(self.conv2(x)))
        
        return x



class Bottleneck(nn.Module):
    """修正后的瓶颈层，确保尺寸不变"""
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.ReLU()
        
        # 时间嵌入处理
        self.time_emb_proj = nn.Sequential(
            nn.Linear(time_emb_dim, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(self, x, t_emb):
        # 第一层卷积
        x = self.act(self.norm(self.conv1(x)))
        
        # 处理时间嵌入
        t_emb = self.time_emb_proj(t_emb)
        t_emb = t_emb.view(t_emb.size(0), t_emb.size(1), 1, 1)
        
        # 添加时间嵌入
        x = x + t_emb
        
        # 第二层卷积
        x = self.act(self.norm(self.conv2(x)))
        
        return x
