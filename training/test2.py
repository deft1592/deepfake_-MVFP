"""
eval pretrained model.
"""
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
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.ff_blend import FFBlendDataset
from dataset.fwa_blend import FWABlendDataset
from dataset.pair_dataset import pairDataset
from PIL import Image
from detectors import DETECTOR
from metrics.base_metrics_class import Recorder
from collections import defaultdict

import argparse
from logger import create_logger
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 * 1024"

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str, 
                    default='./training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument('--weights_path', type=str, default=None)
parser.add_argument("--local_rank", type=int, default=-1)  # 分布式训练/测试使用的local_rank
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化分布式环境
def init_distributed():
    # 如果手动指定了local_rank，则使用指定的值；否则从环境变量中获取
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return device

device = init_distributed()  # 初始化分布式，并设置当前设备

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
        
        # 使用分布式采样器
        sampler = torch.utils.data.distributed.DistributedSampler(
            test_set,
            num_replicas=dist.get_world_size(),
            rank=args.local_rank,
            shuffle=False
        )
        test_data_loader = \
            torch.utils.data.DataLoader(
                dataset=test_set, 
                batch_size=config['test_batchSize'],
                num_workers=int(config['workers']),
                collate_fn=test_set.collate_fn,
                sampler=sampler,  # 关键修改：使用分布式采样器
                pin_memory=True,  # 启用内存锁页，加速数据传输
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)

    return test_data_loaders

def ts2image(tensor):
    """
    将PyTorch张量（可包含负值）转换为PIL图像对象
    
    Returns:
        PIL.Image对象
    """
    # 确保张量在CPU上
    tensor=tensor[0]
    tensor = tensor.cpu().detach()
    
    # 移除批次维度（如果存在）
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # 检查张量维度并调整通道顺序 (C, H, W) -> (H, W, C) 以供PIL处理
    if tensor.dim() == 3:
        if tensor.size(0) in [1, 3]:  # 标准图像通道数
            tensor = tensor.permute(1, 2, 0)
    elif tensor.dim() != 2:
        raise ValueError(f"不支持的张量维度: {tensor.dim()}")
    
    # 关键：处理数值范围（包括负值）
    t_min, t_max = tensor.min(), tensor.max()
    
    # 防止除以零（当张量所有值相等时）
    if t_max - t_min < 1e-6:
        normalized_tensor = torch.zeros_like(tensor)
    else:
        normalized_tensor = (tensor - t_min) / (t_max - t_min)  # 归一化到[0,1]
    
    # 缩放到0-255并转换为uint8
    image_array = (normalized_tensor * 255).to(torch.uint8).numpy()
    
    # 确定图像模式
    if image_array.ndim == 3 and image_array.shape[-1] == 3:
        mode = 'RGB'
    elif image_array.ndim == 3 and image_array.shape[-1] == 1:
        mode = 'L'
        image_array = image_array.squeeze(-1)
    elif image_array.ndim == 2:
        mode = 'L'
    else:
        raise ValueError(f"无法推断图像模式，数组形状: {image_array.shape}")
    
    # 创建PIL图像
    image = Image.fromarray(image_array, mode=mode)
    return image  
    
def test_one_dataset(model, data_loader):
    prediction_lists = []
    #feature_lists = []
    label_lists = []
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            # 1. 空值安全处理 - 关键修复
            data = data_dict['image']
            label = torch.where(data_dict['label'] != 0, 1, 0)

            # 2. 移至GPU（带空值检查）
            data_dict['image'] = data.to(device, non_blocking=True)
            data_dict['label'] = label.to(device, non_blocking=True)
            
            # 修复点：检查mask/landmark是否存在且非空
            if 'mask' in data_dict and data_dict['mask'] is not None:  # 空值检查[3](@ref)
                data_dict['mask'] = data_dict['mask'].to(device, non_blocking=True)
            else:
                data_dict['mask'] = None  # 显式设为None
                
            if 'landmark' in data_dict and data_dict['landmark'] is not None:  # 空值检查[7](@ref)
                data_dict['landmark'] = data_dict['landmark'].to(device, non_blocking=True)
            else:
                data_dict['landmark'] = None  # 显式设为None

            # 3. 模型推理
            predictions = model(data_dict, inference=True)
            
            predictions=predictions['pred_dict']

            # 4. 收集结果（保持在GPU）
            prediction_lists.append(predictions['prob'].detach())

            label_lists.append(data_dict['label'].detach())
            
            
    
    # DistributedSampler assigns non-contiguous dataset indices to each rank.
    # Gather these indices together with predictions so paths stay aligned.
    dataset_indices = torch.tensor(
        list(data_loader.sampler), dtype=torch.long, device=device
    )

    # 4. 合并批次结果（仍在 GPU）
    if not prediction_lists:
        raise RuntimeError(
            f"Dataset loader is empty: {len(data_loader.dataset)} samples"
        )
    prediction_lists = torch.cat(prediction_lists, dim=0)  # GPU 张量
    label_lists = torch.cat(label_lists, dim=0)
    #feature_lists = torch.cat(#feature_lists, dim=0)

    if not (
        prediction_lists.size(0)
        == label_lists.size(0)
        == dataset_indices.size(0)
    ):
        raise RuntimeError(
            "Local result lengths do not match: "
            f"pred={prediction_lists.size(0)}, label={label_lists.size(0)}, "
            f"indices={dataset_indices.size(0)}"
        )
    
    # 5. 准备 all_gather 的接收缓冲区（必须是 CUDA 张量）
    world_size = dist.get_world_size()
    pred_gather = [torch.zeros_like(prediction_lists).to(device) for _ in range(world_size)]
    label_gather = [torch.zeros_like(label_lists).to(device) for _ in range(world_size)]
    index_gather = [torch.zeros_like(dataset_indices) for _ in range(world_size)]
    #feat_gather = [torch.zeros_like(feature_lists).to(device) for _ in range(world_size)]
    
    # 6. 执行 all_gather（确保所有张量在 GPU）
    dist.all_gather(pred_gather, prediction_lists)  # 输入输出均为 CUDA 张量
    dist.all_gather(label_gather, label_lists)
    dist.all_gather(index_gather, dataset_indices)
    #dist.all_gather(feat_gather, feature_lists)
    
    # 7. 主进程聚合结果（移动到 CPU）
    if args.local_rank == 0:
        pred_gather = torch.cat(pred_gather, dim=0).cpu().numpy()
        label_gather = torch.cat(label_gather, dim=0).cpu().numpy()
        index_gather = torch.cat(index_gather, dim=0).cpu().numpy()

        # DistributedSampler may pad with repeated indices when the dataset
        # size is not divisible by the world size. Keep each sample once.
        _, unique_positions = np.unique(index_gather, return_index=True)
        unique_positions = np.sort(unique_positions)
        pred_gather = pred_gather[unique_positions]
        label_gather = label_gather[unique_positions]
        index_gather = index_gather[unique_positions]

        image_paths = [
            data_loader.dataset.data_dict['image'][int(index)]
            for index in index_gather
        ]
        #feat_gather = torch.cat(feat_gather, dim=0).cpu().numpy()
        return pred_gather, label_gather, None, image_paths
    else:
        return None, None, None, None

def test_epoch(model, test_data_loaders):
    model.eval()
    metrics_all_datasets = {}

    for key in test_data_loaders.keys():
        predictions_nps, label_nps, feat_nps, img_names = test_one_dataset(
            model, test_data_loaders[key]
        )
        
        if args.local_rank == 0 and predictions_nps is not None:
            metric_one_dataset = get_test_metrics(
                y_pred=predictions_nps, 
                y_true=label_nps,
                img_names=img_names
            )
            metrics_all_datasets[key] = metric_one_dataset
            
            # 输出结果
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
        # 处理多卡训练保存的权重（移除module前缀）
        state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if args.local_rank == 0:
            print(f'Loaded checkpoint from {weights_path}')
            if missing_keys:
                print(f'Missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'Unexpected keys: {unexpected_keys}')
    else:
        if args.local_rank == 0:
            print('No pretrained weights provided')
    
    # 包装为DDP模型
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    
    # 开始测试
    if args.local_rank == 0:
        print('===> Starting distributed testing')
    metrics = test_epoch(model, test_data_loaders)
    
    # 清理分布式环境
    dist.destroy_process_group()
    
    if args.local_rank == 0:
        print('===> Test Done!')
        return metrics

if __name__ == '__main__':
    main()
