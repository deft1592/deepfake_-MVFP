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
from dataset.deft_ff_dataset import MainDataset
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
parser.add_argument('--output_dir', type=str, default='./tsne_data', help='输出pkl文件的目录')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_filename_without_extension(filepath):
    # 获取文件名（包括后缀）
    filename = os.path.basename(filepath)
    # 分割文件名和扩展名
    name, extension = os.path.splitext(filename)
    return name
# 初始化分布式环境
def init_distributed():
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
        test_set = MainDataset(
                config=config,
            )
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
                sampler=sampler,
                pin_memory=True,
                drop_last=False
            )
        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config['test_dataset']:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders


def save_tsne_data(features, labels, dataset_name, output_dir ,weights_path):
    """保存特征和标签为pkl文件，用于t-SNE可视化"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建保存的数据结构
    tsne_data = {
        'features': features,  # 特征向量
        'labels': labels,      # 对应的标签
        'dataset': dataset_name, # 数据集名称
    }
    
    # 生成文件名
    filename = f"tsne_{dataset_name}_{weights_path}_train.pkl"
    filepath = os.path.join(output_dir, filename)
    
    # 保存为pkl文件
    with open(filepath, 'wb') as f:
        pickle.dump(tsne_data, f)
    
    print(f"t-SNE数据已保存到: {filepath}")
    print(f"特征形状: {features.shape}, 标签形状: {labels.shape}")
    
    return filepath

def test_one_dataset(model, data_loader):
    feature_lists = []
    label_lists = []

    with torch.no_grad():
        for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            data = data_dict['image'].to(device, non_blocking=True)
            label = data_dict['label'].to(device, non_blocking=True)

            data_dict['image'] = data
            data_dict['label'] = label
            data_dict['mask'] = None
            data_dict['landmark'] = None

            predictions = model(data_dict, inference=True)

            feature_lists.append(predictions['feat'].detach().cpu())
            label_lists.append(label.detach().cpu())

    feature_lists = torch.cat(feature_lists, dim=0).numpy()
    label_lists = torch.cat(label_lists, dim=0).numpy()

    gathered_feat = [None for _ in range(dist.get_world_size())]
    gathered_label = [None for _ in range(dist.get_world_size())]

    dist.all_gather_object(gathered_feat, feature_lists)
    dist.all_gather_object(gathered_label, label_lists)

    if args.local_rank == 0:
        return (
            np.concatenate(gathered_label, axis=0),
            np.concatenate(gathered_feat, axis=0),
        )
    else:
        return None, None



def test_epoch(model, test_data_loaders,weights_path):
    model.eval()
    metrics_all_datasets = {}
    
    for key in test_data_loaders.keys():
        
        label_nps, feat_nps = test_one_dataset(model, test_data_loaders[key])
        # 对普通数据集，不使用标签字典
        if args.local_rank == 0:
            save_tsne_data(feat_nps, label_nps, key, args.output_dir,get_filename_without_extension(weights_path))   
    
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
        print(f't-SNE数据将保存到: {args.output_dir}')
    
    metrics = test_epoch(model, test_data_loaders,get_filename_without_extension(weights_path))
    
    # 清理分布式环境
    dist.destroy_process_group()
    
    if args.local_rank == 0:
        print('===> Test Done!')
        return metrics

if __name__ == '__main__':
    main()
