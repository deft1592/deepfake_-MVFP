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
import json
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024 * 1024"

parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_path', type=str,
                    default='./training/config/detector/resnet34.yaml',
                    help='path to detector YAML file')
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument(
    "--test_data_split",
    choices=("val", "test"),
    default=None,
    help="override the dataset split selected by the detector config",
)
parser.add_argument(
    "--gan_dataset_root",
    type=str,
    default=None,
    help=(
        "test every immediate subdirectory of this path as an independent GAN "
        "dataset; each dataset must contain 0_real and 1_fake directories"
    ),
)
parser.add_argument('--weights_path', type=str, default=None)
parser.add_argument(
    '--dataset_json_folder',
    type=str,
    default=None,
    help='override the directory containing dataset JSON index files',
)
parser.add_argument(
    '--output_dir',
    type=str,
    default='.',
    help='root directory for prediction CSV and classification JSON outputs',
)
parser.add_argument('--perturbation_scale', type=float, default=None)
parser.add_argument('--lmdb', action='store_true', help='evaluate from configured LMDB datasets')
parser.add_argument("--local_rank", type=int, default=-1)  # 分布式训练/测试使用的local_rank
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _merge_config(base_config, override_config):
    """Recursively merge an experiment override into its base config."""
    merged = dict(base_config)
    for key, value in override_config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_detector_config(config_path):
    """Load a detector config, including the base_config experiment chain."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    base_config_path = config.pop('base_config', None)
    if base_config_path is None:
        return config
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(
            os.path.dirname(os.path.abspath(config_path)), base_config_path
        )
    return _merge_config(load_detector_config(base_config_path), config)

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
        # These datasets are ordinary image directories. In particular, the
        # configured FF++ LMDB contains c23 but no c40 keys. Other datasets in
        # the same invocation may still use LMDB independently.
        if (
            test_name == 'WDF'
            or test_name.startswith('GAN_')
            or test_name.endswith('_c40')
        ):
            config['lmdb'] = False
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

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
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
            prediction_dict = predictions.get('pred_dict', predictions)

            # 4. 收集结果（保持在GPU）
            prediction_lists.append(prediction_dict['prob'].detach())
            #feature_lists.append(predictions['feat'].detach())
            label_lists.append(data_dict['label'].detach())

    # 4a. 获取 DistributedSampler 分配给本rank的索引（shuffle=False 保证确定性）
    dataset_indices = torch.tensor(list(data_loader.sampler), dtype=torch.long, device=device)

    # 4b. 合并批次结果（仍在 GPU）
    if not prediction_lists:
        raise RuntimeError(f"Dataset loader is empty: {len(data_loader.dataset)} samples")
    prediction_lists = torch.cat(prediction_lists, dim=0)  # GPU 张量
    label_lists = torch.cat(label_lists, dim=0)
    #feature_lists = torch.cat(#feature_lists, dim=0)

    # 长度一致性检查
    assert prediction_lists.size(0) == dataset_indices.size(0) == label_lists.size(0), \
        f'长度不一致: pred={prediction_lists.size(0)}, indices={dataset_indices.size(0)}, label={label_lists.size(0)}'

    # 5. 准备 all_gather 的接收缓冲区（必须是 CUDA 张量）
    world_size = dist.get_world_size()
    pred_gather = [torch.zeros_like(prediction_lists).to(device) for _ in range(world_size)]
    label_gather = [torch.zeros_like(label_lists).to(device) for _ in range(world_size)]
    idx_gather = [torch.zeros_like(dataset_indices).to(device) for _ in range(world_size)]
    #feat_gather = [torch.zeros_like(feature_lists).to(device) for _ in range(world_size)]

    # 6. 执行 all_gather（确保所有张量在 GPU）
    dist.all_gather(pred_gather, prediction_lists)  # 输入输出均为 CUDA 张量
    dist.all_gather(label_gather, label_lists)
    dist.all_gather(idx_gather, dataset_indices)
    #dist.all_gather(feat_gather, feature_lists)

    # 7. 主进程聚合结果（移动到 CPU），并从索引还原图片路径
    if args.local_rank == 0:
        pred_gather = torch.cat(pred_gather, dim=0).cpu().numpy()
        label_gather = torch.cat(label_gather, dim=0).cpu().numpy()
        idx_gather = torch.cat(idx_gather, dim=0).cpu().numpy()

        # DistributedSampler pads the last batch on some ranks. Remove those
        # duplicate indices before calculating metrics.
        _, unique_positions = np.unique(idx_gather, return_index=True)
        unique_positions = np.sort(unique_positions)
        pred_gather = pred_gather[unique_positions]
        label_gather = label_gather[unique_positions]
        idx_gather = idx_gather[unique_positions]
        #feat_gather = torch.cat(feat_gather, dim=0).cpu().numpy()

        # 从聚合索引还原图片路径
        dataset = data_loader.dataset
        img_path_lists = [dataset.data_dict['image'][int(i)] for i in idx_gather]

        return pred_gather, label_gather, None, img_path_lists
    else:
        return None, None, None, None

def save_classification_results(img_names, y_pred, y_true, dataset_name, output_dir="./results"):
    """
    将分类正确和错误的图片路径及标签分别保存为JSON文件
    """
    if args.local_rank != 0:
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 将预测概率转换为类别 (0.5为阈值)
    y_pred_class = (y_pred > 0.5).astype(int)
    y_true = np.clip(y_true, 0, 1).astype(int)

    # 分离正确和错误的分类
    correct_indices = y_pred_class == y_true
    incorrect_indices = ~correct_indices

    # 准备正确分类的数据
    correct_results = []
    for idx in np.where(correct_indices)[0]:
        correct_results.append({
            "image_path": str(img_names[idx]),
            "true_label": int(y_true[idx]),
            "predicted_label": int(y_pred_class[idx]),
            "predicted_prob": float(y_pred[idx])
        })

    # 准备错误分类的数据
    incorrect_results = []
    for idx in np.where(incorrect_indices)[0]:
        incorrect_results.append({
            "image_path": str(img_names[idx]),
            "true_label": int(y_true[idx]),
            "predicted_label": int(y_pred_class[idx]),
            "predicted_prob": float(y_pred[idx])
        })

    # 保存为JSON文件
    correct_file = os.path.join(output_dir, f"{dataset_name}_correct.json")
    incorrect_file = os.path.join(output_dir, f"{dataset_name}_incorrect.json")

    with open(correct_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": dataset_name,
            "total_correct": len(correct_results),
            "results": correct_results
        }, f, ensure_ascii=False, indent=2)

    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": dataset_name,
            "total_incorrect": len(incorrect_results),
            "results": incorrect_results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n分类结果已保存:")
    print(f"  - 正确分类: {correct_file} (共 {len(correct_results)} 张)")
    print(f"  - 错误分类: {incorrect_file} (共 {len(incorrect_results)} 张)")


def test_epoch(model, test_data_loaders):
    model.eval()
    metrics_all_datasets = {}

    if args.local_rank == 0:
        os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "classification_results"), exist_ok=True)

    for key in test_data_loaders.keys():
        predictions_nps, label_nps, feat_nps, img_path_lists = test_one_dataset(model, test_data_loaders[key])

        if args.local_rank == 0 and predictions_nps is not None:
            # img_path_lists 已通过 all_gather + 索引还原，保证与 pred/label 长度一致
            img_names = img_path_lists

            metric_one_dataset = get_test_metrics(
                y_pred=predictions_nps,
                y_true=label_nps,
                img_names=img_names,
                save_path=os.path.join(args.output_dir, "results", f"{key}_pred.csv")
            )
            metrics_all_datasets[key] = metric_one_dataset

            # 保存分类正确/错误的结果
            save_classification_results(
                img_names=img_names,
                y_pred=predictions_nps,
                y_true=label_nps,
                dataset_name=key,
                output_dir=os.path.join(args.output_dir, "classification_results")
            )

            # 输出结果
            tqdm.write(f"dataset: {key}")
            for k, v in metric_one_dataset.items():
                if k not in ['pred', 'label']:  # 不输出预测数组
                    tqdm.write(f"{k}: {v}")

    return metrics_all_datasets

def print_metrics_summary(metrics_all_datasets):
    """Print one compact scalar-metric table after all datasets finish."""
    if args.local_rank != 0:
        return
    print("\n===> Final metrics for all datasets")
    metric_names = ("acc", "auc", "eer", "ap", "video_auc")
    print("dataset\t" + "\t".join(metric_names))
    for dataset_name, metrics in metrics_all_datasets.items():
        values = []
        for name in metric_names:
            value = metrics.get(name, float("nan"))
            values.append(f"{float(value):.6f}" if np.isscalar(value) else "nan")
        print(dataset_name + "\t" + "\t".join(values))

def main():
    # 加载配置
    config = load_detector_config(args.detector_path)
    with open('./training/config/test_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config.update(config2)
    if args.lmdb:
        config['lmdb'] = True
        config['lmdb_dir'] = './datasets/lmdb'
    if args.dataset_json_folder is not None:
        config['dataset_json_folder'] = args.dataset_json_folder

    # 参数覆盖
    requested_datasets = list(args.test_dataset or [])
    if args.gan_dataset_root:
        gan_root = os.path.abspath(os.path.expanduser(args.gan_dataset_root))
        if not os.path.isdir(gan_root):
            parser.error(f"GAN dataset root does not exist: {gan_root}")
        # With explicit GAN_* names, evaluate only those requested datasets.
        # Otherwise discover every immediate subdirectory for convenience.
        explicit_gan_datasets = [
            name for name in requested_datasets if name.startswith("GAN_")
        ]
        gan_datasets = explicit_gan_datasets or [
            f"GAN_{name}"
            for name in sorted(os.listdir(gan_root))
            if os.path.isdir(os.path.join(gan_root, name))
        ]
        if not gan_datasets:
            parser.error(f"No dataset directories found in: {gan_root}")
        config['gan_dataset_root'] = gan_root
        requested_datasets.extend(gan_datasets)
        if args.local_rank == 0:
            print("Discovered GAN datasets: " + ", ".join(gan_datasets))
    if requested_datasets:
        # Preserve order while preventing duplicate evaluation/output names.
        config['test_dataset'] = list(dict.fromkeys(requested_datasets))
    if args.test_data_split is not None:
        config['test_data_split'] = args.test_data_split
    weights_path = args.weights_path or config.get('weights_path')
    if args.perturbation_scale is not None:
        if args.perturbation_scale < 0:
            parser.error("perturbation_scale must be non-negative")
        config['perturbation_scale'] = args.perturbation_scale
    if args.local_rank == 0:
        print(f"Perturbation scale: {float(config.get('perturbation_scale', 1.0))}")

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
        model_state_dict = model.state_dict()
        # disabled_anchor is a non-trainable scalar added solely so the
        # empty-discriminator ablation has an optimizer parameter. Older
        # checkpoints may not contain it, while newer ones may contain it for
        # every variant. Reconcile only this inert compatibility key and keep
        # strict validation for all learned model parameters.
        anchor_suffix = '.disabled_anchor'
        for key in model_state_dict:
            if key.endswith(anchor_suffix) and key not in state_dict:
                state_dict[key] = model_state_dict[key]
        for key in list(state_dict):
            if key.endswith(anchor_suffix) and key not in model_state_dict:
                del state_dict[key]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                f'Checkpoint does not match the configured model: '
                f'missing={missing_keys}, unexpected={unexpected_keys}'
            )
        if args.local_rank == 0:
            print(f'Loaded checkpoint from {weights_path}')
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
        print_metrics_summary(metrics)
        print('===> Test Done!')
        return metrics

if __name__ == '__main__':
    main()
