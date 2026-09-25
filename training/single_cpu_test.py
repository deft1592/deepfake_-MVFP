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

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        config = config.copy()
        config['test_dataset'] = test_name
        test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode='test', 
            )
        
        # 使用普通数据加载器，移除分布式采样器
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                shuffle=False,  # 测试时不需要shuffle
                pin_memory=True,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)

    return test_data_loaders

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
    
def test_one_dataset(model, data_loader):
    prediction_lists = []
    label_lists = []
    img_name_lists = []  # 收集图像名称
    
    model.eval()
    with torch.no_grad():
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
           
            data = data_dict['image']
            label = data_dict['label']
            
            # 收集图像名称
            if 'image_name' in data_dict:
                img_name_lists.extend(data_dict['image_name'])
            elif 'image_path' in data_dict:
                img_name_lists.extend([os.path.basename(path) for path in data_dict['image_path']])
            else:
                img_name_lists.extend([f"img_{i}_{j}" for j in range(data.shape[0])])
            
            data = data.to(device)
            label = label.to(device)
            
            # 简化数据字典
            data_dict_simple = {
                'image': data,
                'label': label
            }
            
            # 添加可选字段
            if 'mask' in data_dict and data_dict['mask'] is not None:
                data_dict_simple['mask'] = data_dict['mask'].to(device)
            if 'landmark' in data_dict and data_dict['landmark'] is not None:
                data_dict_simple['landmark'] = data_dict['landmark'].to(device)

            predictions = model(data_dict_simple, inference=True)
            
            predictions=predictions['pred_dict']
            prediction_lists.append(predictions['prob'].detach().cpu())
            label_lists.append(label.detach().cpu())
    
    # 单卡直接合并结果
    prediction_lists = torch.cat(prediction_lists, dim=0).numpy()
    label_lists = torch.cat(label_lists, dim=0).numpy()
    
    return prediction_lists, label_lists, None

def test_epoch(model, test_data_loaders):
    model.eval()
    metrics_all_datasets = {}

    for key in test_data_loaders.keys():
        data_dict = test_data_loaders[key].dataset.data_dict
        predictions_nps, label_nps, _ = test_one_dataset(model, test_data_loaders[key])
        
        if predictions_nps is not None:
            metric_one_dataset = get_test_metrics(
                y_pred=predictions_nps, 
                y_true=label_nps,
                img_names=data_dict['image']
            )
            metrics_all_datasets[key] = metric_one_dataset
            
            tqdm.write(f"dataset: {key}")
            for k, v in metric_one_dataset.items():
                tqdm.write(f"{k}: {v}")
    
    return metrics_all_datasets

def main():
    # 加载配置
    with open(args.detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    
    # 参数覆盖
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    weights_path = args.weights_path or config.get('weights_path')
    
    # 初始化随机种子
    init_seed(config)
    
    # 准备测试数据加载器
    test_data_loaders = prepare_testing_data(config)
    
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
    
    # 不使用DDP包装
    print('===> Starting single GPU testing')
    metrics = test_epoch(model, test_data_loaders)
    
    print('===> Test Done!')
    return metrics

if __name__ == '__main__':
    main()
