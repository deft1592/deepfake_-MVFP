import os
import numpy as np
from os.path import join
import cv2
import random
import datetime
import time
import yaml
import pickle
from tqdm import tqdm
from copy import deepcopy
from PIL import Image as pil_image
from metrics.utils import get_test_metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
from PIL import Image

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from detectors import DETECTOR

import argparse
from logger import create_logger


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='./training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, default=None)
# 移除 local_rank 参数
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_seed(config):
    if config['manualSeed'] is None:
        config['manualSeed'] = random.randint(1, 10000)
    random.seed(config['manualSeed'])
    torch.manual_seed(config['manualSeed'])
    if config['cuda']:
        torch.cuda.manual_seed_all(config['manualSeed'])


def ts2image(tensor):
    """保持不变"""
    tensor=tensor[0]
    tensor = tensor.cpu().detach()
    
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    if tensor.dim() == 3:
        if tensor.size(0) in [1, 3]:
            tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() != 2:
        raise ValueError(f"不支持的张量维度: {tensor.dim()}")
    
    t_min, t_max = tensor.min(), tensor.max()
    
    if t_max - t_min < 1e-6:
        normalized_tensor = torch.zeros_like(tensor)
    else:
        normalized_tensor = (tensor - t_min) / (t_max - t_min)
    
    image_array = (normalized_tensor * 255).to(torch.uint8).numpy()
    
    if image_array.ndim == 3 and image_array.shape[-1] == 3:
        mode = 'RGB'
    elif image_array.ndim == 3 and image_array.shape[-1] == 1:
        mode = 'L'
        image_array = image_array.squeeze(-1)
    elif image_array.ndim == 2:
        mode = 'L'
    else:
        raise ValueError(f"无法推断图像模式，数组形状: {image_array.shape}")
    
    image = Image.fromarray(image_array, mode=mode)
    return image  
 

def load_image_as_tensor(img_path, image_size=256):
    """
    读取单张图片并转换为模型输入 tensor
    """
    img = Image.open(img_path).convert('RGB')
    img = img.resize((image_size, image_size))

    img = np.array(img).astype(np.float32) / 255.0  # [0,1]
    img = torch.from_numpy(img).permute(2, 0, 1)    # (H,W,C) -> (C,H,W)
    img = img.unsqueeze(0)                          # (1,C,H,W)

    return img


def main():
    # 加载配置
    with open('./training/config/detector/frepdd.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    
    weights_path = 'weights2/frepdd_fft_vggDcgan_multiDis.pth'
    
    # 初始化随机种子
    init_seed(config)
    
    # 准备模型
    model_class = DETECTOR[config['model_name']]
    model = model_class(config).to(device)
    
    # 加载预训练权重
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        # 处理可能的权重格式
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
            
        # 移除可能的module前缀
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        print(f'Loaded checkpoint from {weights_path}')
        if missing_keys:
            print(f'Missing keys: {missing_keys}')
        if unexpected_keys:
            print(f'Unexpected keys: {unexpected_keys}')
    else:
        print('No pretrained weights provided')
    
    

    model.eval()

    
    image_dir="datasets/rgb/Ntire_dataset/valid"

    # 打开txt文件用于写入probabilities
    with open('submission.txt', 'w') as f:
        for input_image_name in os.listdir(image_dir):   
            # 构建完整图片路径
            input_image_path = os.path.join(image_dir, input_image_name)
            # 2️⃣ 读取图片
            input_tensor = load_image_as_tensor(
                input_image_path,
                image_size=config.get('image_size', 256)
            ).to(device)

            # 3️⃣ 前向传播
            with torch.no_grad():
                input={}
                input['image']=input_tensor
                input['label']=torch.tensor([0]).to(device)  
                output = model(input,inference=True)
                prob=output["pred_dict"]['prob']
                # 将prob转换为标量值并写入文件，每行一个
                prob_value = prob.item()  # 假设prob是标量张量
                f.write(f"{prob_value:.1f}\n")
    
   


if __name__ == '__main__':
    main()
