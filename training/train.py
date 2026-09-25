# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.
import multiprocessing


import os
import math
import argparse
from os.path import join
import cv2
import random
import datetime
import time
import yaml
from tqdm import tqdm
import numpy as np
from datetime import timedelta
from copy import deepcopy
from PIL import Image as pil_image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import *
from metrics.utils import parse_metric_for_print
from logger import create_logger, RankFilter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description="Process some paths.")
parser.add_argument(
    "--detector_path",
    type=str,
    default=".v2/training/config/detector/sbi.yaml",
    help="path to detector YAML file",
)

parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument(
    "--weights_path",
    type=str,
    default=None,
    help="model checkpoint used to continue fine-tuning",
)
parser.add_argument(
    "--start_epoch",
    type=int,
    default=None,
    help="epoch number assigned to the first resumed epoch",
)
parser.add_argument(
    "--no-save_ckpt", dest="save_ckpt", action="store_false", default=True
)
parser.add_argument(
    "--no-save_feat", dest="save_feat", action="store_false", default=True
)
parser.add_argument("--ddp", action="store_true", default=False)
parser.add_argument(
    "--task_target",
    type=str,
    default="",
    help="specify the target of current training task",
)
parser.add_argument(
    "--selector_checkpoint",
    type=str,
    default=None,
    help="optional pre-trained selector checkpoint used by FreqDebias Fo-Mixup",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="number of epochs (converted to DeepfakeBench's inclusive final epoch)",
)
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=None,
    help="per-process training batch size override",
)

local_rank = int(os.environ['LOCAL_RANK'])

args = parser.parse_args()

torch.cuda.set_device(local_rank)
# 反归一化函数
def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def show_images(images, title="Samples"):
    grid = make_grid(denormalize(images), nrow=4)
    npimg = grid.numpy()
    plt.figure(figsize=(15, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis("off")
    plt.savefig(
        'output.png',
        bbox_inches='tight',  # 去除白边
        pad_inches=0,        # 内边距为0
        dpi=300)  
    plt.close()
    
def init_seed(config):
    if config["manualSeed"] is None:
        config["manualSeed"] = random.randint(1, 10000)
    random.seed(config["manualSeed"])
    if config["cuda"]:
        torch.manual_seed(config["manualSeed"])
        torch.cuda.manual_seed_all(config["manualSeed"])

def prepare_training_data(config):
    # Only use the blending dataset class in training
    if "dataset_type" in config and config["dataset_type"] == "blend":
        if config["model_name"] == "facexray":
            train_set = FFBlendDataset(config)
        elif config["model_name"] == "fwa":
            train_set = FWABlendDataset(config)
        elif config["model_name"] == "sbi" :
            train_set = SBIDataset(config, mode="train")
        elif config["model_name"] == "lsda":
            train_set = LSDADataset(config, mode="train")
        elif config['model_name']=="deft":
            train_set = DEFTDataset(config, mode="train")
            # 获取一个batch的数据
        else:
            raise NotImplementedError(
                "Only facexray, fwa, sbi, and lsda are currently supported for blending dataset"
            )
    elif "dataset_type" in config and config["dataset_type"] == "pair":
        train_set = pairDataset(
            config, mode="train"
        )  # Only use the pair dataset class in training
    elif "dataset_type" in config and config["dataset_type"] == "iid":
        train_set = IIDDataset(config, mode="train")
    elif "dataset_type" in config and config["dataset_type"] == "I2G":
        train_set = I2GDataset(config, mode="train")
    elif "dataset_type" in config and config["dataset_type"] == "lrl":
        train_set = LRLDataset(config, mode="train")
    else:
        train_set = DeepfakeAbstractBaseDataset(
            config=config,
            mode="train",
        )
    if config["model_name"] == "lsda":
        from dataset.lsda_dataset import CustomSampler

        custom_sampler = CustomSampler(
            num_groups=train_set.num_groups,
            n_frame_per_vid=config["frame_num"]["train"],
            batch_size=config["train_batchSize"],
            videos_per_group=5,
        )
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            num_workers=int(config["workers"]),
            sampler=custom_sampler,
            collate_fn=train_set.collate_fn,
        )
    elif config["ddp"]:
        sampler = DistributedSampler(train_set)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            num_workers=int(config["workers"]),
            collate_fn=train_set.collate_fn,
            sampler=sampler,
        )
    else:
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config["train_batchSize"],
            shuffle=True,
            num_workers=int(config["workers"]),
            collate_fn=train_set.collate_fn,
        )
        
    return train_data_loader

def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = (
            config.copy()
        )  # create a copy of config to avoid altering the original one
        config["test_dataset"] = test_name  # specify the current test dataset
        if config.get("dataset_type", None) == "lrl":
            
            test_set = LRLDataset(
                config=config,
                mode="test",
            )
            
        else:
            test_set = DeepfakeAbstractBaseDataset(
                config=config,
                mode="test",
            )

        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set,
            batch_size=config["test_batchSize"],
            shuffle=False,
            num_workers=int(config["workers"]),
            collate_fn=test_set.collate_fn,
            drop_last=(test_name == "DeepFakeDetection"),
        )

        return test_data_loader

    test_data_loaders = {}
    for one_test_name in config["test_dataset"]:
        test_data_loaders[one_test_name] = get_test_data_loader(config, one_test_name)
    return test_data_loaders

def choose_optimizer(model, config):
    
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name].get('amsgrad', False),
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer



def choose_scheduler(config, optimizer, total_training_steps=None):
    if config["lr_scheduler"] is None:
        return None
    elif config["lr_scheduler"] == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_step"],
            gamma=config["lr_gamma"],
        )
        return scheduler
    elif config["lr_scheduler"] == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["lr_T_max"],
            eta_min=config["lr_eta_min"],
        )
        return scheduler
    elif config["lr_scheduler"] == "cosine_warmup":
        if total_training_steps is None or total_training_steps < 1:
            raise ValueError("total_training_steps is required for cosine_warmup")
        warmup_steps = int(config.get("warmup_steps", 0))
        if warmup_steps >= total_training_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be smaller than "
                f"total_training_steps ({total_training_steps})"
            )
        min_lr_ratio = float(config.get("min_lr_ratio", 0.0))

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_training_steps - warmup_steps)
            )
            cosine_ratio = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_ratio

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config["lr_scheduler"] == "linear":
        scheduler = LinearDecayLR(
            optimizer,
            config["nEpochs"],
            int(config["nEpochs"] / 4),
        )
    else:
        raise NotImplementedError(
            "Scheduler {} is not implemented".format(config["lr_scheduler"])
        )


def choose_metric(config):
    metric_scoring = config["metric_scoring"]
    if metric_scoring not in ["eer", "auc", "acc", "ap"]:
        raise NotImplementedError("metric {} is not implemented".format(metric_scoring))
    return metric_scoring


def main():
    
    #multiprocessing.set_start_method('spawn', force=True)
    # parse options and load config
    with open(args.detector_path, "r") as f:
        config = yaml.safe_load(f)
    with open("./training/config/train_config.yaml", "r") as f:
        config2 = yaml.safe_load(f)
    if "label_dict" in config:
        config2["label_dict"] = config["label_dict"]
    config.update(config2)
    config["local_rank"] = int(os.environ["LOCAL_RANK"])
    if config["dry_run"]:
        config["nEpochs"] = 0
        config["save_feat"] = False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config["train_dataset"] = args.train_dataset
    if args.test_dataset:
        config["test_dataset"] = args.test_dataset
    if args.selector_checkpoint:
        config["selector_checkpoint"] = args.selector_checkpoint
    if args.weights_path:
        config["weights_path"] = args.weights_path
    if args.start_epoch is not None:
        if args.start_epoch < 0:
            raise ValueError("--start_epoch must be non-negative")
        config["start_epoch"] = args.start_epoch
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        config["nEpochs"] = config["start_epoch"] + args.epochs - 1
    if args.train_batch_size is not None:
        if args.train_batch_size < 1:
            raise ValueError("--train_batch_size must be at least 1")
        config["train_batchSize"] = args.train_batch_size
    config["save_ckpt"] = args.save_ckpt
    config["save_feat"] = args.save_feat
    if config["lmdb"]:
        config["dataset_json_folder"] = "preprocessing/dataset_json_v3"
    # create logger
    timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task_str = (
        f"_{config['task_target']}"
        if config.get("task_target", None) is not None
        else ""
    )
    logger_path = os.path.join(
        config["log_dir"], config["model_name"] + task_str + "_" + timenow
    )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, "training.log"))
    logger.info("Save log to {}".format(logger_path))
    config["ddp"] = args.ddp
    # print configuration
    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters: \n"
    for key, value in config.items():
        params_string += "{}: {}".format(key, value) + "\n"
    logger.info(params_string)

    # init seed
    init_seed(config)


    # set cudnn benchmark if needed
    if config["cudnn"]:
        cudnn.benchmark = True
    if config["ddp"]:
        # dist.init_process_group(backend='nccl')
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=40))
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)

     # prepare the model (detector)
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)

    weights_path = config.get("weights_path")
    if weights_path:
        weights_path = os.path.expanduser(weights_path)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        checkpoint = torch.load(weights_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Unsupported checkpoint format at {weights_path}: "
                f"expected a state dict, got {type(checkpoint).__name__}"
            )
        state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()
        }
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded model weights from %s", weights_path)
        if missing_keys:
            logger.warning("Missing checkpoint keys: %s", missing_keys)
        if unexpected_keys:
            logger.warning("Unexpected checkpoint keys: %s", unexpected_keys)

    # prepare the optimizer
    optimizer = choose_optimizer(model, config)

    # prepare the scheduler
    total_epochs = config["nEpochs"] - config["start_epoch"] + 1
    total_training_steps = len(train_data_loader) * total_epochs
    scheduler = choose_scheduler(config, optimizer, total_training_steps)

    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(
        config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow
    )

    # start training
    for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
        trainer.model.epoch = epoch
        
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            test_data_loaders=test_data_loaders,
        )
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with testing {metric_scoring}: {parse_metric_for_print(best_metric)}!"
            )
        # update
        if "svdd" in config["model_name"]:
            model.update_R(epoch)
        if scheduler is not None and config["lr_scheduler"] != "cosine_warmup":
            scheduler.step()
    logger.info(
        "Stop Training on best Testing metric {}".format(
            parse_metric_for_print(best_metric)
        )
    )

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
