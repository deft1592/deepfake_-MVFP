# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer
import os
import sys
from torchvision.utils import make_grid
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics
from PIL import Image

FFpp_pool = ["FaceForensics++", "FF-DF", "FF-F2F", "FF-FS", "FF-NT"]  #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FREPDD_DEBUG_DIR = "./images/frepdd_img"


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

class Trainer(object):
    def __init__(
            self,
            config,
            model,
            optimizer,
            scheduler,
            logger,
            metric_scoring="auc",
            time_now=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            swa_model=None,
        ):
            if config is None or model is None or optimizer is None or logger is None:
                raise ValueError(
                    "config, model, optimizer, logger, and tensorboard writer must be implemented"
                )

            self.config = config
            self.model = model
            self.optimizer = optimizer
            self.optimg, self.optimd, self.optimc = self.optimizer
            self.scheduler = scheduler
            if self.scheduler is None:
                self.scheduler = []
            elif not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]

            self.swa_model = swa_model
            self.writers = {}  # TensorBoard writers
            self.logger = logger
            self.metric_scoring = metric_scoring
            self.best_ckpt_path = None
            self.last_avg_improved = False

            # maintain the best metric of all epochs
            self.best_metrics_all_time = defaultdict(
                lambda: defaultdict(
                    lambda: float("-inf") if self.metric_scoring != "eer" else float("inf")
                )
            )
            self.speed_up()  # move model to GPU

            # create directory path
            self.timenow = time_now
            if "task_target" not in config:
                self.log_dir = os.path.join(
                    self.config["log_dir"], self.config["model_name"] + "_" + self.timenow
                )
            else:
                task_str = f"_{config['task_target']}" if config["task_target"] is not None else ""
                self.log_dir = os.path.join(
                    self.config["log_dir"], self.config["model_name"] + task_str + "_" + self.timenow
                )
            os.makedirs(self.log_dir, exist_ok=True)


    def update_lam(self, epoch, max_epoch):
        t = epoch / max_epoch

        lam_mag    = 1.0 - 0.5 * t
        lam_radial = 0.5 + 0.7 * t
        lam_band   = 0.3
        lam_phase  = 0.2 * (1 - t)

        # 直接写入 buffer（DDP-safe）
        self.model.module.lam_mag.fill_(lam_mag)
        self.model.module.lam_radial.fill_(lam_radial)
        self.model.module.lam_band.fill_(lam_band)
        self.model.module.lam_phase.fill_(lam_phase)


    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir, phase, dataset_key, metric_key, "metric_board"
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config["ddp"] == True:
            num_gpus = torch.cuda.device_count()
            
            #self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            #local_rank=[i for i in range(0,num_gpus)]
            self.model = DDP(
                self.model,
                device_ids=[self.config["local_rank"]],
                find_unused_parameters=True,
                output_device=self.config["local_rank"],
            )
            
    def calculate_spectrum_stats(self, x: torch.Tensor) -> dict:
        """
        Calculate the spectrum stats (mean and std) of the image tensor x.
        """
        x_fft = torch.fft.fft2(x, norm='ortho')  # FFT transformation
        x_fft_shift = torch.fft.fftshift(x_fft, dim=(-2, -1))  # Shift the spectrum to the center

        magnitude = torch.abs(x_fft_shift)  # Compute the magnitude of the spectrum

        mean_mag = torch.mean(magnitude)  # Mean magnitude
        std_mag = torch.std(magnitude)    # Standard deviation of magnitude

        return {"mean": mean_mag.item(), "std": std_mag.item()}

    def log_spectrum_stats_to_tensorboard(self,writer, epoch, real_images, perturbed_images):
        """
        Log the spectrum stats to TensorBoard.
        """
        real_stats = self.calculate_spectrum_stats(real_images)
        perturbed_stats = self.calculate_spectrum_stats(perturbed_images)

        # Log to TensorBoard
        writer.add_scalar('Spectrum/Real Image Mean', real_stats['mean'], epoch)
        writer.add_scalar('Spectrum/Real Image Std', real_stats['std'], epoch)
        writer.add_scalar('Spectrum/Perturbed Image Mean', perturbed_stats['mean'], epoch)
        writer.add_scalar('Spectrum/Perturbed Image Std', perturbed_stats['std'], epoch)

    def plot_spectrum(self, writer,images: torch.Tensor, title: str, epoch: int):
        """
        Plot the spectrum of the images and save it to TensorBoard.
        """
        B, C, H, W = images.shape
        fft_images = torch.fft.fft2(images, norm='ortho')
        fft_images_shifted = torch.fft.fftshift(fft_images, dim=(-2, -1))

        magnitude = torch.abs(fft_images_shifted)
        magnitude_log = torch.log(magnitude + 1e-8)  # Log transform for better visualization

        grid = make_grid(magnitude_log, nrow=4)

        npimg = grid.cpu().numpy()
        # 如果需要将数值范围规范化到 [0, 1]，可以在此进行处理
        npimg = npimg / npimg.max()  # 将图像值归一化到 [0, 1]

        # 转换维度顺序为 (C, H, W)
        npimg = np.transpose(npimg, (1, 2, 0))

        # Record to TensorBoard
        img_tensor = torch.from_numpy(npimg).permute(2, 0, 1).float()  # 转换为 [C, H, W]
        writer.add_image(title, img_tensor, epoch)

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location="cpu")
            suffix = model_path.split(".")[-1]
            if suffix == "p":
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info("Model found in {}".format(model_path))
        else:
            raise NotImplementedError("=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config["ddp"] == True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if "svdd" in self.config["model_name"]:
                torch.save(
                    {
                        "R": self.model.R,
                        "c": self.model.c,
                        "state_dict": self.model.state_dict(),
                    },
                    save_path,
                )
            else:
                torch.save(self.model.state_dict(), save_path)
        self.logger.info(
            f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}"
        )
        self.best_ckpt_path = save_path

    def step_schedulers(self):
        for scheduler in self.scheduler:
            if scheduler is not None:
                scheduler.step()

    def detector(self):
        return self.model.module if type(self.model) is DDP else self.model

    def is_main_process(self):
        return not (
            self.config["ddp"] and dist.is_initialized() and dist.get_rank() != 0
        )

    def set_requires_grad(self, module, flag):
        for param in module.parameters():
            param.requires_grad = flag

    def has_discriminator(self):
        return bool(self.detector().enabled_discriminator_branches)

    def apply_backbone_train_scope(self, scope):
        if scope == "all":
            return
        backbone = self.detector().backbone
        self.set_requires_grad(backbone, False)
        for name, parameter in backbone.named_parameters():
            is_head = name.startswith("last_layer.")
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
            is_last_stage = name.startswith(efficientnet_last_stage_prefixes) or name.startswith(
                vit_last_stage_prefixes
            )
            if is_head or (scope == "last_stage" and is_last_stage):
                parameter.requires_grad = True

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = fea
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        if not self.is_main_process():
            return
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f"data_dict_{phase}.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, "metric_dict_best.pickle")
        with open(file_path, "wb") as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_step(self, iteration, data_dict):
        train_mode = self.config.get("train_mode", "all")
        net = self.detector()

        # Mode: Classifier only
        if train_mode == "classifier":
            self.optimc.zero_grad()
            self.apply_backbone_train_scope(self.config.get("backbone_train_scope", "all"))
            outputs = self.model(data_dict, for_classifier=True)
            cls_loss = net.get_cls_loss(data_dict, outputs)
            cls_loss.backward()
            self.optimc.step()

            return {
                "cls_loss": cls_loss.detach(),
                "g_loss": torch.tensor(0.0, device=cls_loss.device),
                "d_loss": torch.tensor(0.0, device=cls_loss.device),
            }, outputs

        # Mode: GAN (G + D)
        elif train_mode == "gan":
            # Train D
            d_loss = torch.tensor(0.0, device=device)

            if self.has_discriminator() and iteration % 1 == 0:
                self.optimd.zero_grad()
                outputs_d = self.model(data_dict, for_discriminator=True)
                d_loss = net.get_d_loss(data_dict, outputs_d)
                d_loss.backward()
                self.optimd.step()

            # Train G
            self.optimg.zero_grad()
            outputs_g = self.model(data_dict, for_generator=True)
            g_loss = net.get_g_loss(data_dict, outputs_g)
            g_loss.backward()
            self.optimg.step()

            return {
                "g_loss": g_loss.detach(),
                "d_loss": d_loss.detach(),
                "cls_loss": torch.tensor(0.0, device=g_loss.device),
            }, outputs_g

        # Stable warm-start: keep D fixed and make G task-aware. This avoids
        # the moving G/D/C target that caused cross-domain forgetting.
        elif train_mode == "task_finetune":
            self.set_requires_grad(net.generator, True)
            self.set_requires_grad(net.discriminator, False)
            self.set_requires_grad(net.backbone, False)
            self.optimg.zero_grad()
            outputs_g = self.model(data_dict, for_generator=True)
            g_loss = net.get_g_loss(data_dict, outputs_g)
            g_loss.backward()
            self.optimg.step()

            self.set_requires_grad(net.generator, False)
            self.set_requires_grad(net.discriminator, False)
            self.set_requires_grad(net.backbone, True)
            self.apply_backbone_train_scope(self.config.get("backbone_train_scope", "all"))
            self.optimc.zero_grad()
            outputs = self.model(data_dict, for_classifier=True)
            cls_loss = net.get_cls_loss(data_dict, outputs)
            cls_loss.backward()
            self.optimc.step()

            self.set_requires_grad(net.generator, True)
            self.set_requires_grad(net.discriminator, False)
            self.set_requires_grad(net.backbone, True)
            return {
                "g_loss": g_loss.detach(),
                "d_loss": torch.tensor(0.0, device=g_loss.device),
                "cls_loss": cls_loss.detach(),
            }, outputs

        # Mode: Joint training (D -> G -> classifier)
        elif train_mode == "all":
            # Train D first so G updates against a fresher discriminator
            d_loss = torch.tensor(0.0, device=device)
            d_update_every = max(1, int(self.config.get("d_update_every", 1)))
            if self.has_discriminator() and iteration % d_update_every == 0:
                self.set_requires_grad(net.generator, False)
                self.set_requires_grad(net.discriminator, True)
                self.set_requires_grad(net.backbone, False)
                self.optimd.zero_grad()
                outputs_d = self.model(data_dict, for_discriminator=True)
                d_loss = net.get_d_loss(data_dict, outputs_d)
                d_loss.backward()
                self.optimd.step()

            # Train G
            self.set_requires_grad(net.generator, True)
            self.set_requires_grad(net.discriminator, False)
            self.set_requires_grad(net.backbone, False)
            self.optimg.zero_grad()
            outputs_g = self.model(data_dict, for_generator=True)
            g_loss = net.get_g_loss(data_dict, outputs_g)
            g_loss.backward()
            self.optimg.step()

            # Train classifier
            self.set_requires_grad(net.generator, False)
            self.set_requires_grad(net.discriminator, False)
            self.set_requires_grad(net.backbone, True)
            self.apply_backbone_train_scope(self.config.get("backbone_train_scope", "all"))
            self.optimc.zero_grad()
            outputs = self.model(data_dict, for_classifier=True)
            cls_loss = net.get_cls_loss(data_dict, outputs)
            cls_loss.backward()
            self.optimc.step()
            self.set_requires_grad(net.generator, True)
            self.set_requires_grad(net.discriminator, True)
            self.set_requires_grad(net.backbone, True)

            return {
                "g_loss": g_loss.detach(),
                "d_loss": d_loss.detach(),
                "cls_loss": cls_loss.detach(),
            }, outputs

        else:
            raise ValueError(f"Unknown train_mode: {train_mode}")

    def train_epoch(
        self,
        epoch,
        max_epoch,
        train_data_loader,
        test_data_loaders=None,
    ):
        # self.update_lam(epoch, max_epoch)
        # print(self.model.module.lam_mag,self.model.module.lam_radial,self.model.module.lam_band,self.model.module.lam_phase)
        self.logger.info("===> Epoch[{}] start!".format(epoch))
        validations_per_epoch = max(
            1, int(self.config.get("validations_per_epoch", 1))
        )
        test_step = max(1, len(train_data_loader) // validations_per_epoch)
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        #print("data_dict: ", data_dict)
        self.save_data_dict("train", data_dict, ",".join(self.config["train_dataset"]))
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm( enumerate(train_data_loader), total=len(train_data_loader)):
            self.setTrain()
            if self.config.get("freeze_backbone_bn", False):
                for module in self.detector().backbone.modules():
                    if isinstance(module, nn.modules.batchnorm._BatchNorm):
                        module.eval()
            # more elegant and more scalable way of moving data to GPU
            for key in data_dict.keys():
                if data_dict[key] != None and key != "name":
                    data_dict[key] = data_dict[key].cuda()

            losses, predictions = self.train_step(iteration,data_dict)

            # Keep the generator statistics before selecting the classifier
            # prediction dictionary. This is detached in the detector and is
            # safe to aggregate with the scalar recorders below.
            pert_rms = predictions.get("freq_data_dict", {}).get("pert_rms")
            predictions=predictions['pred_dict']
            if pert_rms is not None:
                train_recorder_loss["pert_rms"].update(pert_rms)

            # Log the spectrum stats to TensorBoard

            # Get real images and perturbed images (or outputs from the model)
            real_images = data_dict["image"]  # Assuming "image" is the key for the input images
            perturbed_images = predictions.get('perturbed_images', real_images)  # or any other output you want to visualize


            if (
                "SWA" in self.config
                and self.config["SWA"]
                and epoch > self.config["swa_start"]
            ):
                self.swa_model.update_parameters(self.model)

            # compute training metric for each batch data
            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(
                    data_dict, predictions
                )
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            # run tensorboard to visualize the training process
            if iteration % 300 == 0 and self.config["local_rank"] == 0:

                writer = self.get_writer(
                        "train", ",".join(self.config["train_dataset"]),""
                    )
                self.log_spectrum_stats_to_tensorboard(writer,epoch, real_images, perturbed_images)

                # Plot and log the spectrum images to TensorBoard
                self.plot_spectrum(writer,real_images, title='Real Image Spectrum', epoch=epoch)
                self.plot_spectrum(writer,perturbed_images, title='Perturbed Image Spectrum', epoch=epoch)

                if self.config["SWA"] and (
                    epoch > self.config["swa_start"] or self.config["dry_run"]
                ):
                    self.step_schedulers()
                # info for loss
                loss_str = f"Iter: {step_cnt}    "

                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg == None:
                        loss_str += f"training-loss, {k}: not calculated"
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    # tensorboard-1. loss
                    writer = self.get_writer(
                        "train", ",".join(self.config["train_dataset"]), k
                    )
                    writer.add_scalar(f"train_loss/{k}", v_avg, global_step=step_cnt)
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg == None:
                        metric_str += f"training-metric, {k}: not calculated    "
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    # tensorboard-2. metric
                    writer = self.get_writer(
                        "train", ",".join(self.config["train_dataset"]), k
                    )
                    writer.add_scalar(f"train_metric/{k}", v_avg, global_step=step_cnt)
                self.logger.info(metric_str)

                # clear recorder.
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for (
                    name,
                    recorder,
                ) in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for (
                    name,
                    recorder,
                ) in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            
            #self.cleanup_epoch_resources(train_data_loader)
            
            # run test
            if (step_cnt + 1) % test_step == 0:
                if test_data_loaders is not None and (not self.config["ddp"]):
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                elif test_data_loaders is not None and self.config["ddp"]:
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch,
                        iteration,
                        test_data_loaders,
                        step_cnt,
                    )
                else:
                    test_best_metric = None

                    # total_end_time = time.time()
            # total_elapsed_time = total_end_time - total_start_time
            # print("总花费的时间: {:.2f} 秒".format(total_elapsed_time))
            step_cnt += 1
         
        
        return test_best_metric



    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = pred == label
        zero_num = len(label) - np.count_nonzero(label)
        acc_fake = np.count_nonzero(judge[zero_num:]) / len(judge[zero_num:])
        acc_real = np.count_nonzero(judge[:zero_num]) / len(judge[:zero_num])
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader):
        # define test recorder
        cnt=0
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        feature_lists = []
        label_lists = []
        for i, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            # get data
            if "label_spe" in data_dict:
                data_dict.pop("label_spe")  # remove the specific label
            data_dict["label"] = torch.where(
                data_dict["label"] != 0, 1, 0
            )  # fix the label to 0 and 1 only
            # move data to GPU elegantly
            for key in data_dict.keys():
                if data_dict[key] != None:
                    data_dict[key] = data_dict[key].cuda()
            # model forward without considering gradient computation
            predictions = self.inference(data_dict)

            if self.is_main_process() and cnt<=5:
                os.makedirs(FREPDD_DEBUG_DIR, exist_ok=True)
                perturbation=ts2image(predictions['perturbation'])
                perturbed_image=ts2image(predictions['perturbed_image'])

                perturbation.save(os.path.join(FREPDD_DEBUG_DIR, f'perturbation_{i}.png'))
                perturbed_image.save(os.path.join(FREPDD_DEBUG_DIR, f'perturbed_image_{i}.png'))
                cnt+=1

            #predictions=predictions['pred_dict']
            label_lists.extend(data_dict['label'].cpu().tolist())
            prediction_lists.extend(predictions['pred_dict']['prob'].cpu().tolist())
            
            #feature_lists += list(predictions["feat"].cpu().detach().numpy())
            if type(self.model) is not AveragedModel:
                # compute all losses for each batch data
                if type(self.model) is DDP:
                    losses = self.model.module.get_cls_loss(data_dict,predictions)
                    #losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                # store data by recorder
                # for name, value in losses.items():
                #     test_recorder_loss[name].update(value)
                    
                test_recorder_loss['cls_loss'].update(losses)

        return (
            test_recorder_loss,
            np.array(prediction_lists),
            np.array(label_lists),
            np.array(feature_lists)
        )

    def save_best(
        self,
        epoch,
        iteration,
        step,
        losses_one_dataset_recorder,
        key,
        metric_one_dataset,
        phase="val",
        save_best_ckpt=True,
    ):
        is_main_process = self.is_main_process()
        best_metric = self.best_metrics_all_time[key].get(
            self.metric_scoring,
            float("-inf") if self.metric_scoring != "eer" else float("inf"),
        )
        # Check if the current score is an improvement
        min_delta = (
            float(self.config.get("early_stop_min_delta", 0.0))
            if key == "avg"
            else 0.0
        )
        improved = (
            (metric_one_dataset[self.metric_scoring] > best_metric + min_delta)
            if self.metric_scoring != "eer"
            else (metric_one_dataset[self.metric_scoring] < best_metric - min_delta)
        )
        if improved:
            # Update the best metric
            self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[
                self.metric_scoring
            ]
            if key == "avg":
                self.last_avg_improved = True
                self.best_metrics_all_time[key]["dataset_dict"] = metric_one_dataset[
                    "dataset_dict"
                ]
            # Save checkpoint, feature, and metrics if specified in config
            if save_best_ckpt and self.config["save_ckpt"]:
                self.best_ckpt_path = os.path.join(self.log_dir, phase, key, "ckpt_best.pth")
                if is_main_process:
                    self.save_ckpt(phase, key, f"{epoch}+{iteration}")
            if is_main_process:
                self.save_metrics(phase, metric_one_dataset, key)
        if not is_main_process:
            return
        if losses_one_dataset_recorder is not None:
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer(phase, key, k)
                v_avg = v.average()
                if v_avg == None:
                    print(f"{k} is not calculated")
                    continue
                # tensorboard-1. loss
                writer.add_scalar(f"test_losses/{k}", v_avg, global_step=step)
                loss_str += f"testing-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)
        # tqdm.write(loss_str)
        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k == "pred" or k == "label" or k == "dataset_dict":
                continue
            metric_str += f"testing-metric, {k}: {v}    "
            # tensorboard-2. metric
            writer = self.get_writer(phase, key, k)
            writer.add_scalar(f"test_metrics/{k}", v, global_step=step)
        if "pred" in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(
                metric_one_dataset["pred"], metric_one_dataset["label"]
            )
            metric_str += f"testing-metric, acc_real:{acc_real}; acc_fake:{acc_fake}"
            writer.add_scalar(f"test_metrics/acc_real", acc_real, global_step=step)
            writer.add_scalar(f"test_metrics/acc_fake", acc_fake, global_step=step)
        self.logger.info(metric_str)

    def test_epoch(self, epoch, iteration, test_data_loaders, step, phase="val", save_best_ckpt=True):
        if self.config["ddp"] and dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()
            return self.best_metrics_all_time

        self.last_avg_improved = False

        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        final_metric_snapshot = {}
        best_metrics_per_dataset = defaultdict(
            dict
        )  # best metric for each dataset, for each metric
        avg_metric = {
            "acc": 0,
            "auc": 0,
            "eer": 0,
            "ap": 0,
            "video_auc": 0,
            "dataset_dict": {},
        }
        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict("test", data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_nps, label_nps, feature_nps = (
                self.test_one_dataset(test_data_loaders[key])
            )
           
            losses_all_datasets[key] = losses_one_dataset_recorder
            metric_one_dataset = get_test_metrics(
                y_pred=predictions_nps, y_true=label_nps, img_names=data_dict["image"]
            )
            final_metric_snapshot[key] = metric_one_dataset
            for metric_name, value in metric_one_dataset.items():
                if metric_name in avg_metric:
                    avg_metric[metric_name] += value
            avg_metric["dataset_dict"][key] = metric_one_dataset[self.metric_scoring]
            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue
            self.save_best(
                epoch,
                iteration,
                step,
                losses_one_dataset_recorder,
                key,
                metric_one_dataset,
                phase=phase,
                # Per-dataset best values are useful diagnostics, but only the
                # macro average is allowed to select the final checkpoint.
                save_best_ckpt=False,
            )

        if len(keys) > 0 and self.config.get("save_avg", False):
            # calculate avg value
            for key in avg_metric:
                if key != "dataset_dict":
                    avg_metric[key] /= len(keys)
            self.save_best(
                epoch,
                iteration,
                step,
                None,
                "avg",
                avg_metric,
                phase=phase,
                save_best_ckpt=save_best_ckpt,
            )

        if phase == "target_test" and self.is_main_process():
            for key, metric_one_dataset in final_metric_snapshot.items():
                self.logger.info(
                    f"[TARGET TEST] {key}: auc={metric_one_dataset.get('auc')}, "
                    f"video_auc={metric_one_dataset.get('video_auc')}, "
                    f"ap={metric_one_dataset.get('ap')}, eer={metric_one_dataset.get('eer')}"
                )
            if avg_metric["dataset_dict"]:
                self.logger.info(
                    f"[TARGET TEST] avg: auc={avg_metric.get('auc')}, "
                    f"video_auc={avg_metric.get('video_auc')}, "
                    f"ap={avg_metric.get('ap')}, eer={avg_metric.get('eer')}"
                )

        if self.config["ddp"] and dist.is_initialized():
            dist.barrier()
        self.logger.info("===> Test Done!")
        return (
            self.best_metrics_all_time
        )  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.detector()(data_dict, inference=True)
        return predictions
