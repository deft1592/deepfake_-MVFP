# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
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

from trainer.trainer_new import Trainer
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
parser.add_argument("--val_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--weights_path", type=str, default=None)
parser.add_argument("--nEpochs", type=int, default=None)
parser.add_argument("--start_epoch", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--lr_d", type=float, default=None)
parser.add_argument("--lr_c", type=float, default=None)
parser.add_argument("--lr_head", type=float, default=None)
parser.add_argument("--perturbation_scale", type=float, default=None)
parser.add_argument("--lr_step", type=int, default=None)
parser.add_argument("--lr_gamma", type=float, default=None)
parser.add_argument("--g_compression_weight", type=float, default=None)
parser.add_argument("--g_adv_weight", type=float, default=None)
parser.add_argument("--g_task_weight", type=float, default=None)
parser.add_argument("--band_dropout_prob", type=float, default=None)
parser.add_argument(
    "--train_mode", choices=("all", "gan", "classifier", "task_finetune")
)
parser.add_argument("--normalize_compression_loss", action="store_true")
parser.add_argument("--validate_before_training", action="store_true")
parser.add_argument("--freeze_backbone_bn", action="store_true")
parser.add_argument("--early_stop_patience", type=int, default=None)
parser.add_argument("--early_stop_min_delta", type=float, default=None)
parser.add_argument("--train_subset_dataset", type=str, default=None)
parser.add_argument("--train_subset_fraction", type=float, default=None)
parser.add_argument("--train_subset_seed", type=int, default=None)
parser.add_argument("--class_weights", type=float, nargs=2, default=None)
parser.add_argument(
    "--backbone_train_scope", choices=("all", "last_stage", "head"), default=None
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

local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
args = parser.parse_args()


def _merge_config(base_config, override_config):
    merged = dict(base_config)
    for key, value in override_config.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_detector_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    base_config_path = config.pop("base_config", None)
    if base_config_path is None:
        return config
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(
            os.path.dirname(os.path.abspath(config_path)),
            base_config_path,
        )
    return _merge_config(load_detector_config(base_config_path), config)


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
        elif config['model_name']=="frepddCg":
            train_data_strategy = config.get("train_data_strategy", "standard")
            if train_data_strategy == "sbi":
                if len(config.get("train_dataset", [])) != 1:
                    raise ValueError("SBI training requires exactly one source dataset")
                train_set = SBIDataset(config, mode="train")
            elif len(config.get("train_dataset", [])) > 1:
                train_set = CombinedTrainDataset(
                    [
                        DeepfakeAbstractBaseDataset(
                            {**config, "train_dataset": [dataset_name]}, mode="train"
                        )
                        for dataset_name in config["train_dataset"]
                    ]
                )
            elif train_data_strategy == "standard":
                train_set = DeepfakeAbstractBaseDataset(config, mode="train")
            else:
                raise ValueError(
                    f"Unsupported frepddCg train_data_strategy: {train_data_strategy}"
                )
            #train_set = FreddDataset(config, mode="train")
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
            num_groups=2 * 360,
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
        
    batch = next(iter(train_data_loader))
            
    # 解析真实和伪造图像
    real_images = batch["image"]
    
    
    show_images(real_images, "Real Images")
    return train_data_loader


def prepare_testing_data(config):
    def get_test_data_loader(config, test_name):
        # update the config dictionary with the specific testing dataset
        config = (
            config.copy()
        )  # create a copy of config to avoid altering the original one
        config["test_dataset"] = test_name  # specify the current test dataset
        # WDF is stored as ordinary image files in this repository, not LMDB.
        if test_name == "WDF":
            config["lmdb"] = False
        if config.get("dataset_type", None) == "lrl":
            
            test_set = LRLDataset(
                config=config,
                mode="test",
            )
            
        elif config['model_name']=="frepddCg":
            test_set = DeepfakeAbstractBaseDataset(config, mode="test")
            #test_set = FreddDataset(config, mode="test")
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


def prepare_eval_data(config, dataset_key):
    eval_config = config.copy()
    eval_config["test_dataset"] = list(eval_config.get(dataset_key, []))
    split_key = dataset_key.replace("_dataset", "_data_split")
    if split_key in eval_config:
        eval_config["test_data_split"] = eval_config[split_key]
    return prepare_testing_data(eval_config)


def choose_optimizer(model, config,model_type):
    
    opt_name = config["optimizer"]["type"]
    if opt_name == "sgd":
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
        )
        return optimizer
    elif opt_name == "adam":
        base_lr = config["optimizer"][opt_name]["lr"]
        learning_rate = config["optimizer"][opt_name].get(
            f"lr_{model_type}", base_lr
        )
        if model_type=="d":
            learning_rate = config["optimizer"][opt_name].get("lr_d", learning_rate)
            optimizer = optim.Adam(
                params=model.parameters(),
                lr=learning_rate / 2,
                weight_decay=config["optimizer"][opt_name]["weight_decay"],
                betas=(
                    config["optimizer"][opt_name]["beta1"],
                    config["optimizer"][opt_name]["beta2"],
                ),
                eps=config["optimizer"][opt_name]["eps"],
                amsgrad=config["optimizer"][opt_name]["amsgrad"],
            )
        else:
            params = model.parameters()
            if model_type == "c" and config["optimizer"][opt_name].get("lr_head") is not None:
                head_lr = config["optimizer"][opt_name]["lr_head"]
                backbone_params = []
                head_params = []
                for name, parameter in model.named_parameters():
                    if name.startswith("last_layer."):
                        head_params.append(parameter)
                    else:
                        backbone_params.append(parameter)
                params = [
                    {"params": backbone_params, "lr": learning_rate},
                    {"params": head_params, "lr": head_lr},
                ]
            optimizer = optim.Adam(
                params=params,
                lr=learning_rate,
                weight_decay=config["optimizer"][opt_name]["weight_decay"],
                betas=(
                    config["optimizer"][opt_name]["beta1"],
                    config["optimizer"][opt_name]["beta2"],
                ),
                eps=config["optimizer"][opt_name]["eps"],
                amsgrad=config["optimizer"][opt_name]["amsgrad"],
            )
        return optimizer
    elif opt_name == "sam":
        optimizer = SAM(
            model.parameters(),
            optim.SGD,
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
        )
    else:
        raise NotImplementedError(
            "Optimizer {} is not implemented".format(config["optimizer"])
        )
        
    return optimizer


def choose_scheduler(config, optimizer):
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

def set_requires_grad(module, flag=True):
    for p in module.parameters():
        p.requires_grad = flag


def _matches_train_scope(name, scope):
    if scope == "all":
        return True
    if name.startswith("last_layer."):
        return True
    if scope == "head":
        return False

    efficientnet_last_stage_prefixes = (
        "efficientnet._blocks.22.",
        "efficientnet._blocks.23.",
        "efficientnet._blocks.24.",
        "efficientnet._blocks.25.",
        "efficientnet._blocks.26.",
        "efficientnet._blocks.27.",
        "efficientnet._blocks.28.",
        "efficientnet._blocks.29.",
        "efficientnet._blocks.30.",
        "efficientnet._blocks.31.",
        "efficientnet._conv_head.",
        "efficientnet._bn1.",
    )
    vit_last_stage_prefixes = (
        "vit.blocks.9.",
        "vit.blocks.10.",
        "vit.blocks.11.",
        "vit.norm.",
        "clip.vision_model.encoder.layers.9.",
        "clip.vision_model.encoder.layers.10.",
        "clip.vision_model.encoder.layers.11.",
        "clip.vision_model.post_layernorm.",
        "clip.vision_model.encoder.layers.20.",
        "clip.vision_model.encoder.layers.21.",
        "clip.vision_model.encoder.layers.22.",
        "clip.vision_model.encoder.layers.23.",
        "clip.vision_model.post_layernorm.",
    )
    return name.startswith(efficientnet_last_stage_prefixes) or name.startswith(
        vit_last_stage_prefixes
    )


def set_train_mode(model, mode, config=None):
    if mode == "classifier":
        # 只训练 classifier
        set_requires_grad(model.generator, False)
        set_requires_grad(model.discriminator, False)
        set_requires_grad(model.backbone, True)
        scope = (config or {}).get("backbone_train_scope", "all")
        if scope != "all":
            set_requires_grad(model.backbone, False)
            for name, parameter in model.backbone.named_parameters():
                if _matches_train_scope(name, scope):
                    parameter.requires_grad = True

    elif mode == "gan":
        # 只训练 G + D
        set_requires_grad(model.generator, True)
        set_requires_grad(model.discriminator, True)
        set_requires_grad(model.backbone, False)

    elif mode in ("all", "task_finetune"):
        set_requires_grad(model, True)

    else:
        raise ValueError(f"Unknown train_mode: {mode}")

def main():
    # parse options and load config
    
    config = load_detector_config(args.detector_path)
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
    if args.val_dataset:
        config["val_dataset"] = args.val_dataset
    elif "val_dataset" not in config:
        config["val_dataset"] = config["train_dataset"]
    if args.weights_path:
        config["weights_path"] = args.weights_path
    if args.nEpochs is not None:
        config["nEpochs"] = args.nEpochs
    # Optional command-line overrides are useful for warm-start fine-tuning
    # without changing the baseline detector YAML.
    adam_config = config.setdefault("optimizer", {}).setdefault("adam", {})
    for arg_name in ("lr", "lr_d", "lr_c", "lr_head"):
        value = getattr(args, arg_name)
        if value is not None:
            adam_config[arg_name] = value
    for arg_name in (
        "lr_step",
        "lr_gamma",
        "g_compression_weight",
        "g_adv_weight",
        "g_task_weight",
        "band_dropout_prob",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            config[arg_name] = value
    if args.train_mode is not None:
        config["train_mode"] = args.train_mode
    if args.normalize_compression_loss:
        config["normalize_compression_loss"] = True
    if args.validate_before_training:
        config["validate_before_training"] = True
    if args.freeze_backbone_bn:
        config["freeze_backbone_bn"] = True
    if args.early_stop_patience is not None:
        config["early_stop_patience"] = max(0, args.early_stop_patience)
    if args.early_stop_min_delta is not None:
        config["early_stop_min_delta"] = max(0.0, args.early_stop_min_delta)
    if args.train_subset_dataset is not None:
        config["train_subset_dataset"] = args.train_subset_dataset
    if args.train_subset_fraction is not None:
        if not 0.0 < args.train_subset_fraction <= 1.0:
            raise ValueError("train_subset_fraction must be in (0, 1].")
        config["train_subset_fraction"] = args.train_subset_fraction
    if args.train_subset_seed is not None:
        config["train_subset_seed"] = args.train_subset_seed
    if args.class_weights is not None:
        config["class_weights"] = args.class_weights
    if args.backbone_train_scope is not None:
        config["backbone_train_scope"] = args.backbone_train_scope
    if args.perturbation_scale is not None:
        if args.perturbation_scale < 0:
            raise ValueError("perturbation_scale must be non-negative")
        config["perturbation_scale"] = args.perturbation_scale
    if args.task_target:
        config["task_target"] = args.task_target
    if args.start_epoch is not None:
        config["start_epoch"] = args.start_epoch
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
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        logger.addFilter(RankFilter(0))
    # prepare the training data loader
    train_data_loader = prepare_training_data(config)

    # Validation selects checkpoints; target tests are evaluated only after training.
    val_data_loaders = prepare_eval_data(config, "val_dataset")
    test_data_loaders = prepare_eval_data(config, "test_dataset")

    # prepare the model (detector)
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)

    weights_path = config.get("weights_path",None)
    # 加载预训练权重
    if weights_path is not None:
        ckpt = torch.load(weights_path)
        # 处理多卡训练保存的权重（移除module前缀）
        state_dict = {k.replace('module.', ''): v for k, v in ckpt.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if local_rank == 0:
            print(f'Loaded checkpoint from {weights_path}')
            if missing_keys:
                print(f'Missing keys: {missing_keys}')
            if unexpected_keys:
                print(f'Unexpected keys: {unexpected_keys}')
    else:
        if local_rank == 0:
            print('No pretrained weights provided')

    modelg=model.generator
    modeld=model.discriminator
    modelc=model.backbone
   

    optimizer=[]
    optimizer1 = choose_optimizer(modelg, config,"g")##generator
    optimizer2 = choose_optimizer(modeld, config,"d")##discriminator
    optimizer3 = choose_optimizer(modelc, config,"c")
    
    optimizer.append(optimizer1)
    optimizer.append(optimizer2)
    optimizer.append(optimizer3)
    
    
    scheduler = [choose_scheduler(config, opt) for opt in optimizer]
    
    set_train_mode(model, config["train_mode"], config)
    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(
        config, model, optimizer, scheduler, logger, metric_scoring, time_now=timenow
    )

    if config.get("validate_before_training", False):
        logger.info("===> Warm-start baseline validation!")
        trainer.test_epoch(
            epoch=-1,
            iteration=0,
            test_data_loaders=val_data_loaders,
            step=0,
        )

    # start training
    epochs_without_improvement = 0
    for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
        trainer.model.epoch = epoch
        if config["ddp"] and hasattr(train_data_loader.sampler, "set_epoch"):
            train_data_loader.sampler.set_epoch(epoch)
        
        best_metric = trainer.train_epoch(
            epoch=epoch,
            max_epoch=config['nEpochs'],
            train_data_loader=train_data_loader,
            test_data_loaders=val_data_loaders,
        )
        trainer.step_schedulers()
        patience = int(config.get("early_stop_patience", 0))
        stop_flag = torch.zeros(
            1, dtype=torch.int32, device=torch.device("cuda", local_rank)
        )
        if local_rank == 0 and patience > 0:
            if trainer.last_avg_improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stop_flag.fill_(1)
        if config["ddp"]:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item():
            logger.info(
                "Early stopping after %d validation epochs without avg %s improvement.",
                epochs_without_improvement,
                metric_scoring,
            )
            break
        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with validation {metric_scoring}: {parse_metric_for_print(best_metric)}!"
            )
    logger.info(
        "Stop Training on best validation metric {}".format(
            parse_metric_for_print(best_metric)
        )
    )
    # update
    if "svdd" in config["model_name"]:
        model.update_R(epoch)
    if getattr(trainer, "best_ckpt_path", None):
        trainer.load_ckpt(trainer.best_ckpt_path)
    if test_data_loaders:
        trainer.test_epoch(
            epoch=epoch,
            iteration=0,
            test_data_loaders=test_data_loaders,
            step=(epoch + 1) * len(train_data_loader),
            phase="target_test",
            save_best_ckpt=False,
        )

    # close the tensorboard writers
    for writer in trainer.writers.values():
        writer.close()


if __name__ == "__main__":
    main()
