# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: Abstract Base Class for all types of deepfake datasets.

import sys
from pathlib import Path

import lmdb

sys.path.append(".")

import os
import math
import hashlib
import yaml
import glob
import json

import numpy as np
from copy import deepcopy
import cv2
import random
from PIL import Image
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T

import albumentations as A

from .albu import IsotropicResize

FFpp_pool = [
    "FaceForensics++",
    "FaceShifter",
    "DeepFakeDetection",
    "FF-DF",
    "FF-F2F",
    "FF-FS",
    "FF-NT",
]  #


def all_in_pool(inputs, pool):
    for each in inputs:
        if each not in pool:
            return False
    return True


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Abstract base class for all deepfake datasets.
    """

    def __init__(self, config=None, mode="train"):
        """Initializes the dataset object.

        Args:
            config (dict): A dictionary containing configuration parameters.
            mode (str): A string indicating the mode (train or test).

        Raises:
            NotImplementedError: If mode is not train or test.
        """

        # Set the configuration and mode
        self.config = config
        self.mode = mode
        self.compression = config["compression"]
        self.frame_num = config["frame_num"][mode]

        # Check if 'video_mode' exists in config, otherwise set video_level to False
        self.video_level = config.get("video_mode", False)
        self.clip_size = config.get("clip_size", None)
        self.lmdb = config.get("lmdb", False)
        # Dataset dictionary
        self.image_list = []
        self.label_list = []

        # Set the dataset dictionary based on the mode
        if mode == "train":
            dataset_list = config["train_dataset"]
            # Training data should be collected together for training
            image_list, label_list = [], []
            for one_data in dataset_list:
                tmp_image, tmp_label, tmp_name = (
                    self.collect_img_and_label_for_one_dataset(one_data)
                )
                image_list.extend(tmp_image)
                label_list.extend(tmp_label)
            if self.lmdb:
                if len(dataset_list) > 1:
                    if all_in_pool(dataset_list, FFpp_pool):
                        lmdb_path = os.path.join(
                            config["lmdb_dir"], f"FaceForensics++_lmdb"
                        )
                        self.env = lmdb.open(
                            lmdb_path,
                            create=False,
                            subdir=True,
                            readonly=True,
                            lock=False,
                        )
                    else:
                        raise ValueError(
                            "Training with multiple dataset and lmdb is not implemented yet."
                        )
                else:
                    lmdb_path = os.path.join(
                        config["lmdb_dir"],
                        f"{dataset_list[0] if dataset_list[0] not in FFpp_pool else 'FaceForensics++'}_lmdb",
                    )
                    self.env = lmdb.open(
                        lmdb_path, create=False, subdir=True, readonly=True, lock=False
                    )
        elif mode == "test":
            one_data = config["test_dataset"]
            # Test dataset should be evaluated separately. So collect only one dataset each time
            if one_data == 'WDF':
                image_list, label_list = self.collect_img_and_label_for_WDF()
            elif one_data=='AIGI':
                image_list, label_list = self.collect_img_and_label_for_AIGI()
            elif one_data=='DIFF':
                image_list, label_list = self.collect_img_and_label_for_Diff()
            elif one_data.startswith("GAN_"):
                image_list, label_list = self.collect_img_and_label_for_GAN(one_data)
            else:
                image_list, label_list, name_list = (
                self.collect_img_and_label_for_one_dataset(one_data)
            )
            if self.lmdb:
                lmdb_dataset_name = one_data
                if one_data.endswith("_c40"):
                    lmdb_dataset_name = one_data[:-4]
                lmdb_path = os.path.join(
                    config["lmdb_dir"],
                    (
                        f"{lmdb_dataset_name}_lmdb"
                        if lmdb_dataset_name not in FFpp_pool
                        else "FaceForensics++_lmdb"
                    ),
                )
                self.env = lmdb.open(
                    lmdb_path, create=False, subdir=True, readonly=True, lock=False
                )
        else:
            raise NotImplementedError("Only train and test modes are supported.")

        if mode == "test" and not self.lmdb:
            image_list, label_list = self.filter_missing_samples(image_list, label_list)

        assert (
            len(image_list) != 0 and len(label_list) != 0
        ), f"Collect nothing for {mode} mode!"
        self.image_list, self.label_list = image_list, label_list

        # Create a dictionary containing the image and label lists
        self.data_dict = {
            "image": self.image_list,
            "label": self.label_list,
        }

        self.transform = self.init_data_aug_method()

    def init_data_aug_method(self):
        aug = self.config["data_aug"]
        trans = A.Compose(
            [
                A.HorizontalFlip(p=aug["flip_prob"]),
                A.Rotate(
                    limit=aug["rotate_limit"],
                    p=aug["rotate_prob"],
                ),
                A.GaussianBlur(
                    blur_limit=aug["blur_limit"],
                    p=aug["blur_prob"],
                ),
                A.OneOf(
                    [
                        IsotropicResize(
                            max_side=self.config["resolution"],
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_CUBIC,
                        ),
                        IsotropicResize(
                            max_side=self.config["resolution"],
                            interpolation_down=cv2.INTER_AREA,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                        IsotropicResize(
                            max_side=self.config["resolution"],
                            interpolation_down=cv2.INTER_LINEAR,
                            interpolation_up=cv2.INTER_LINEAR,
                        ),
                    ],
                    p=0 if self.config["with_landmark"] else 1,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=aug["brightness_limit"],
                            contrast_limit=aug["contrast_limit"],
                        ),
                        A.FancyPCA(),
                        A.HueSaturationValue(),
                    ],
                    p=aug.get("color_prob", 0.5),
                ),
                # Simulate unknown social-media and capture pipelines. One
                # degradation is sampled at a time to avoid destroying cues.
                A.OneOf(
                    [
                        A.Downscale(
                            scale_min=aug.get("scale_min", 0.5),
                            scale_max=aug.get("scale_max", 0.9),
                            interpolation=cv2.INTER_AREA,
                        ),
                        A.GaussNoise(
                            var_limit=tuple(aug.get("noise_var_limit", [5.0, 30.0]))
                        ),
                        A.ImageCompression(
                            quality_lower=aug["quality_lower"],
                            quality_upper=aug["quality_upper"],
                        ),
                    ],
                    p=aug.get("degradation_prob", 0.5),
                ),
                A.RandomGamma(
                    gamma_limit=tuple(aug.get("gamma_limit", [85, 115])),
                    p=aug.get("gamma_prob", 0.15),
                ),
                A.ToGray(p=aug.get("grayscale_prob", 0.05)),
            ],
            keypoint_params=(
                A.KeypointParams(format="xy") if self.config["with_landmark"] else None
            ),
        )
        return trans

    def rescale_landmarks(self, landmarks, original_size=256, new_size=224):
        scale_factor = new_size / original_size
        rescaled_landmarks = landmarks * scale_factor
        return rescaled_landmarks

    def collect_img_and_label_for_one_dataset(self, dataset_name: str):
        """Collects image and label lists.

        Args:
            dataset_name (str): A list containing one dataset information. e.g., 'FF-F2F'

        Returns:
            list: A list of image paths.
            list: A list of labels.

        Raises:
            ValueError: If image paths or labels are not found.
            NotImplementedError: If the dataset is not implemented yet.
        """
        # Initialize the label and frame path lists
        label_list = []
        frame_path_list = []

        # Record video name for video-level metrics
        video_name_list = []

        # Compression aliases share the base dataset JSON and LMDB.
        requested_dataset_name = dataset_name
        cp = None
        if dataset_name.endswith("_c40"):
            base_name = dataset_name[:-4]
            if base_name in ("FaceForensics++", "FF-DF", "FF-F2F", "FF-FS", "FF-NT"):
                dataset_name = base_name
                cp = "c40"

        # Try to get the dataset information from the JSON file
        if not os.path.exists(self.config["dataset_json_folder"]):
            self.config["dataset_json_folder"] = self.config[
                "dataset_json_folder"
            ].replace("/Youtu_Pangu_Security_Public", "/Youtu_Pangu_Security/public")
        try:
            with open(
                os.path.join(
                    self.config["dataset_json_folder"], dataset_name + ".json"
                ),
                "r",
            ) as f:
                dataset_info = json.load(f)
        except Exception as e:
            print(e)
            raise ValueError(f"dataset {requested_dataset_name} not exist!")

        # If JSON file exists, do the following data collection
        # FIXME: ugly, need to be modified here.
        # Get the information for the current dataset
        for label in dataset_info[dataset_name]:
            data_split = (
                self.config.get("test_data_split", "test")
                if self.mode == "test"
                else self.mode
            )
            sub_dataset_info = dataset_info[dataset_name][label][data_split]
            # Special case for FaceForensics++ and DeepFakeDetection, choose the compression type
            if cp == None and dataset_name in [
                "FF-DF",
                "FF-F2F",
                "FF-FS",
                "FF-NT",
                "FaceForensics++",
                "DeepFakeDetection",
                "FaceShifter",
            ]:
                sub_dataset_info = sub_dataset_info[self.compression]
            elif cp == "c40" and dataset_name in [
                "FF-DF",
                "FF-F2F",
                "FF-FS",
                "FF-NT",
                "FaceForensics++",
                "DeepFakeDetection",
                "FaceShifter",
            ]:
                sub_dataset_info = sub_dataset_info["c40"]

            if not isinstance(sub_dataset_info, dict):
                raise ValueError(
                    f"Invalid dataset JSON structure in {dataset_name}.json: "
                    f"label={label!r}, split={data_split!r} must be a mapping of "
                    "video names to records."
                )
            if "label" in sub_dataset_info and "frames" in sub_dataset_info:
                raise ValueError(
                    f"Invalid dataset JSON structure in {dataset_name}.json: "
                    f"label={label!r}, split={data_split!r} contains a single video "
                    "record instead of a mapping. Use --test_data_split test or "
                    "regenerate this split."
                )

            subset_dataset = self.config.get("train_subset_dataset")
            subset_fraction = float(self.config.get("train_subset_fraction", 1.0))
            if (
                self.mode == "train"
                and dataset_name == subset_dataset
                and 0.0 < subset_fraction < 1.0
            ):
                subset_seed = int(self.config.get("train_subset_seed", 1024))
                video_names = sorted(
                    sub_dataset_info,
                    key=lambda name: hashlib.sha256(
                        f"{subset_seed}:{dataset_name}:{label}:{name}".encode("utf-8")
                    ).digest(),
                )
                keep_count = max(1, int(round(len(video_names) * subset_fraction)))
                selected = set(video_names[:keep_count])
                sub_dataset_info = {
                    name: info
                    for name, info in sub_dataset_info.items()
                    if name in selected
                }

            # Iterate over the videos in the dataset
            for video_name, video_info in sub_dataset_info.items():
                if not isinstance(video_info, dict):
                    raise ValueError(
                        f"Invalid video record in {dataset_name}.json: "
                        f"label={label!r}, split={data_split!r}, "
                        f"video={video_name!r}, got {type(video_info).__name__}."
                    )
                # Unique video name
                unique_video_name = video_info["label"] + "_" + video_name

                # Get the label and frame paths for the current video
                if video_info["label"] not in self.config["label_dict"]:
                    raise ValueError(
                        f'Label {video_info["label"]} is not found in the configuration file.'
                    )
                label = self.config["label_dict"][video_info["label"]]
                frame_paths = video_info["frames"]
                # sorted video path to the lists
                if "\\" in frame_paths[0]:
                    frame_paths = sorted(
                        frame_paths, key=lambda x: int(x.split("\\")[-1].split(".")[0])
                    )
                else:
                    frame_paths = sorted(
                        frame_paths, key=lambda x: int(x.split("/")[-1].split(".")[0])
                    )

                # Consider the case when the actual number of frames (e.g., 270) is larger than the specified (i.e., self.frame_num=32)
                # In this case, we select self.frame_num frames from the original 270 frames
                total_frames = len(frame_paths)
                if self.frame_num < total_frames:
                    total_frames = self.frame_num
                    if self.video_level:
                        # Select clip_size continuous frames
                        start_frame = (
                            random.randint(0, total_frames - self.frame_num)
                            if self.mode == "train"
                            else 0
                        )
                        frame_paths = frame_paths[
                            start_frame : start_frame + self.frame_num
                        ]  # update total_frames
                    else:
                        # Select self.frame_num frames evenly distributed throughout the video
                        step = total_frames // self.frame_num
                        frame_paths = [
                            frame_paths[i] for i in range(0, total_frames, step)
                        ][: self.frame_num]

                # If video-level methods, crop clips from the selected frames if needed
                if self.video_level:
                    if self.clip_size is None:
                        raise ValueError(
                            "clip_size must be specified when video_level is True."
                        )
                    # Check if the number of total frames is greater than or equal to clip_size
                    if total_frames >= self.clip_size:
                        # Initialize an empty list to store the selected continuous frames
                        selected_clips = []

                        # Calculate the number of clips to select
                        num_clips = total_frames // self.clip_size

                        if num_clips > 1:
                            # Calculate the step size between each clip
                            clip_step = (total_frames - self.clip_size) // (
                                num_clips - 1
                            )

                            # Select clip_size continuous frames from each part of the video
                            for i in range(num_clips):
                                # Ensure start_frame + self.clip_size - 1 does not exceed the index of the last frame
                                start_frame = (
                                    random.randrange(
                                        i * clip_step,
                                        min(
                                            (i + 1) * clip_step,
                                            total_frames - self.clip_size + 1,
                                        ),
                                    )
                                    if self.mode == "train"
                                    else i * clip_step
                                )
                                continuous_frames = frame_paths[
                                    start_frame : start_frame + self.clip_size
                                ]
                                assert (
                                    len(continuous_frames) == self.clip_size
                                ), "clip_size is not equal to the length of frame_path_list"
                                selected_clips.append(continuous_frames)

                        else:
                            start_frame = (
                                random.randrange(0, total_frames - self.clip_size + 1)
                                if self.mode == "train"
                                else 0
                            )
                            continuous_frames = frame_paths[
                                start_frame : start_frame + self.clip_size
                            ]
                            assert (
                                len(continuous_frames) == self.clip_size
                            ), "clip_size is not equal to the length of frame_path_list"
                            selected_clips.append(continuous_frames)

                        # Append the list of selected clips and append the label
                        label_list.extend([label] * len(selected_clips))
                        frame_path_list.extend(selected_clips)
                        # video name save
                        video_name_list.extend(
                            [unique_video_name] * len(selected_clips)
                        )

                    else:
                        print(
                            f"Skipping video {unique_video_name} because it has less than clip_size ({self.clip_size}) frames ({total_frames})."
                        )

                # Otherwise, extend the label and frame paths to the lists according to the number of frames
                else:
                    # Extend the label and frame paths to the lists according to the number of frames
                    label_list.extend([label] * total_frames)
                    frame_path_list.extend(frame_paths)
                    # video name save
                    video_name_list.extend([unique_video_name] * len(frame_paths))

        # Shuffle the label and frame path lists in the same order
        shuffled = list(zip(label_list, frame_path_list, video_name_list))
        random.shuffle(shuffled)
        label_list, frame_path_list, video_name_list = zip(*shuffled)

        return frame_path_list, label_list, video_name_list




    def collect_img_and_label_for_GAN(self, one_data):
        """Collect one GAN dataset from ``0_real``/``1_fake`` directories.

        A dataset may contain the label directories directly (for example
        ``biggan/0_real``), or below category directories (for example
        ``progan/car/0_real``).  The root is supplied by the test entry point,
        keeping this dataset loader portable across machines.
        """
        # Keep compatibility with the project's Python 3.8 environment.
        dataset_name = one_data[len("GAN_"):]
        dataset_root = Path(
            self.config.get(
                "gan_dataset_root",
                Path(__file__).resolve().parents[2] / "GAN_generated_dataset",
            )
        ).expanduser().resolve()
        dataset_dir = dataset_root / dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"GAN dataset directory does not exist: {dataset_dir}"
            )

        image_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp",
            ".jfif", ".pjpeg", ".pjp",
        }
        frame_path_list = []
        label_list = []
        label_counts = {}

        for label_dir_name, label in (("0_real", 0), ("1_fake", 1)):
            label_dirs = sorted(
                path for path in dataset_dir.rglob(label_dir_name) if path.is_dir()
            )
            paths = sorted(
                str(path.resolve())
                for label_dir in label_dirs
                for path in label_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in image_extensions
            )
            frame_path_list.extend(paths)
            label_list.extend([label] * len(paths))
            label_counts[label_dir_name] = len(paths)

        if not all(label_counts.values()):
            raise ValueError(
                f"GAN dataset '{dataset_name}' must contain readable images under "
                f"0_real and 1_fake; found {label_counts} in {dataset_dir}"
            )

        print(
            f"Loaded GAN dataset {dataset_name}: "
            f"{label_counts['0_real']} real, {label_counts['1_fake']} fake"
        )
        return frame_path_list, label_list

    def collect_img_and_label_for_AIGI(self):
        # 定义两个路径
        real_path = "/media/buu/f9ac1451-f2c8-4f9b-ac35-02ca205e1120/deft_work_folder/DeepfakeBench/datasets/rgb/UADFV/real/frames"
        fake_path = "/media/buu/f9ac1451-f2c8-4f9b-ac35-02ca205e1120/deft_work_folder/DeepfakeBench/datasets/rgb/generated.photos"
        
        # 支持的图片扩展名
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', 
                        '.tiff', '.tif', '.webp', '.svg', '.ico',
                        '.jfif', '.pjpeg', '.pjp', '.avif', '.apng'}
        
        # 初始化列表
        frame_path_list = []
        label_list = []
        
        # 处理真实图片 (label=0)
        real_path_obj = Path(real_path)
        if real_path_obj.exists() and real_path_obj.is_dir():
            print(f"正在处理真实图片路径: {real_path}")
            # 递归遍历所有文件
            for file_path in real_path_obj.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    frame_path_list.append(str(file_path))
                    label_list.append(0)
        else:
            print(f"警告: 路径不存在或不是目录: {real_path}")
        
        # 处理伪造图片 (label=1)
        fake_path_obj = Path(fake_path)
        if fake_path_obj.exists() and fake_path_obj.is_dir():
            print(f"正在处理伪造图片路径: {fake_path}")
            # 递归遍历所有文件
            for file_path in fake_path_obj.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    frame_path_list.append(str(file_path))
                    label_list.append(1)
        else:
            print(f"警告: 路径不存在或不是目录: {fake_path}")
        
        return frame_path_list, label_list


    def collect_img_and_label_for_Diff(self):
        # 定义两个路径
        real_path = "/media/buu/f9ac1451-f2c8-4f9b-ac35-02ca205e1120/deft_work_folder/DeepfakeBench/datasets/rgb/UADFV/real/frames"
        fake_path = "/media/buu/f9ac1451-f2c8-4f9b-ac35-02ca205e1120/deft_work_folder/DeepfakeBench/datasets/rgb/Midjourney"
        
        # 支持的图片扩展名
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', 
                        '.tiff', '.tif', '.webp', '.svg', '.ico',
                        '.jfif', '.pjpeg', '.pjp', '.avif', '.apng'}
        
        # 初始化列表
        frame_path_list = []
        label_list = []
        
        # 处理真实图片 (label=0)
        real_path_obj = Path(real_path)
        if real_path_obj.exists() and real_path_obj.is_dir():
            print(f"正在处理真实图片路径: {real_path}")
            # 递归遍历所有文件
            for file_path in real_path_obj.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    frame_path_list.append(str(file_path))
                    label_list.append(0)
        else:
            print(f"警告: 路径不存在或不是目录: {real_path}")
        
        # 处理伪造图片 (label=1)
        fake_path_obj = Path(fake_path)
        if fake_path_obj.exists() and fake_path_obj.is_dir():
            print(f"正在处理伪造图片路径: {fake_path}")
            # 递归遍历所有文件
            for file_path in fake_path_obj.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    frame_path_list.append(str(file_path))
                    label_list.append(1)
        else:
            print(f"警告: 路径不存在或不是目录: {fake_path}")
        
        return frame_path_list, label_list


    def collect_img_and_label_for_WDF(self):
        # The aaai27 snapshot lives one level below the repository data root.
        # Also accept an aaai27-local WDF directory for portable snapshots.
        aaai27_root = Path(__file__).resolve().parents[2]
        wdf_roots = (aaai27_root / "WDF", aaai27_root.parent / "WDF")
        wdf_root = next((path for path in wdf_roots if path.is_dir()), wdf_roots[0])
        real_path = wdf_root / "real_test" / "images"
        fake_path = wdf_root / "fake_test" / "images"
        
        # 支持的图片扩展名
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', 
                        '.tiff', '.tif', '.webp', '.svg', '.ico',
                        '.jfif', '.pjpeg', '.pjp', '.avif', '.apng'}
        
        # 初始化列表
        frame_path_list = []
        label_list = []
        
        # 处理真实图片 (label=0)
        real_path_obj = Path(real_path)
        if real_path_obj.exists() and real_path_obj.is_dir():
            print(f"正在处理真实图片路径: {real_path}")
            # 递归遍历所有文件
            for file_path in real_path_obj.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    frame_path_list.append(str(file_path))
                    label_list.append(0)
        else:
            print(f"警告: 路径不存在或不是目录: {real_path}")
        
        # 处理伪造图片 (label=1)
        fake_path_obj = Path(fake_path)
        if fake_path_obj.exists() and fake_path_obj.is_dir():
            print(f"正在处理伪造图片路径: {fake_path}")
            # 递归遍历所有文件
            for file_path in fake_path_obj.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                    frame_path_list.append(str(file_path))
                    label_list.append(1)
        else:
            print(f"警告: 路径不存在或不是目录: {fake_path}")
        
        return frame_path_list, label_list

       
        
    def resolve_rgb_path(self, file_path):
        """Resolve paths from dataset JSON files across supported data roots."""
        raw_path = os.fspath(file_path)
        normalized_path = raw_path.replace("\\", os.sep)
        if os.path.isabs(normalized_path):
            candidates = [normalized_path]
        else:
            repository_root = Path(__file__).resolve().parents[3]
            candidates = [
                normalized_path,
                os.path.join(self.config["rgb_dir"], normalized_path),
                os.path.join(os.getcwd(), normalized_path),
                os.path.join(os.fspath(repository_root), normalized_path),
            ]
        return next((path for path in candidates if os.path.exists(path)), None), candidates

    def filter_missing_samples(self, image_list, label_list):
        """Remove missing test samples before distributed sampling begins."""
        valid_images = []
        valid_labels = []
        missing_count = 0
        for image_paths, label in zip(image_list, label_list):
            paths = image_paths if isinstance(image_paths, list) else [image_paths]
            if all(self.resolve_rgb_path(path)[0] is not None for path in paths):
                valid_images.append(image_paths)
                valid_labels.append(label)
            else:
                missing_count += 1
        if missing_count:
            print(f"Skipped {missing_count} missing test samples before evaluation")
        return valid_images, valid_labels

    def load_rgb(self, file_path):
        """
        Load an RGB image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the image file.

        Returns:
            An Image object containing the loaded and resized image.

        Raises:
            ValueError: If the loaded image is None.
        """
        size = self.config[
            "resolution"
        ]  # if self.mode == "train" else self.config['resolution']
        if not self.lmdb:
            # Dataset JSON files use Windows separators and not all datasets
            # share the same root: c40 uses ``./dataset``, while the normal
            # preprocessed datasets use ``./datasets/rgb``.
            raw_path = os.fspath(file_path)
            resolved_path, candidates = self.resolve_rgb_path(file_path)
            if resolved_path is None:
                raise FileNotFoundError(
                    f"Image does not exist: {raw_path}; checked {candidates}"
                )
            img = cv2.imread(resolved_path)
            if img is None:
                raise ValueError("Loaded image is None: {}".format(resolved_path))
        elif self.lmdb:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0] == ".":
                    file_path = file_path.replace("./datasets\\", "").replace("\\", "/")

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    raise ValueError(f"LMDB key '{file_path}' not found or data is empty.")
                image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Failed to decode image for key '{file_path}'.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.array(img, dtype=np.uint8))

    def load_mask(self, file_path):
        """
        Load a binary mask image from a file path and resize it to a specified resolution.

        Args:
            file_path: A string indicating the path to the mask file.

        Returns:
            A numpy array containing the loaded and resized mask.

        Raises:
            None.
        """
        size = self.config["resolution"]
        if file_path is None:
            return np.zeros((size, size, 1))
        if not self.lmdb:
            if not file_path[0] == ".":
                file_path = f'./{self.config["rgb_dir"]}\\' + file_path
            if os.path.exists(file_path):
                mask = cv2.imread(file_path, 0)
                if mask is None:
                    mask = np.zeros((size, size))
            else:
                return np.zeros((size, size, 1))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0] == ".":
                    file_path = file_path.replace("./datasets\\", "")

                image_bin = txn.get(file_path.encode())
                if image_bin is None:
                    mask = np.zeros((size, size, 3))
                else:
                    image_buf = np.frombuffer(image_bin, dtype=np.uint8)
                    # cv2.IMREAD_GRAYSCALE为灰度图，cv2.IMREAD_COLOR为彩色图
                    mask = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        mask = cv2.resize(mask, (size, size)) / 255
        mask = np.expand_dims(mask, axis=2)
        return np.float32(mask)

    def load_landmark(self, file_path):
        """
        Load 2D facial landmarks from a file path.

        Args:
            file_path: A string indicating the path to the landmark file.

        Returns:
            A numpy array containing the loaded landmarks.

        Raises:
            None.
        """
        if file_path is None:
            return np.zeros((81, 2))
        if not self.lmdb:
            if not file_path[0] == ".":
                file_path = f'./{self.config["rgb_dir"]}\\' + file_path
            if os.path.exists(file_path):
                landmark = np.load(file_path)
            else:
                return np.zeros((81, 2))
        else:
            with self.env.begin(write=False) as txn:
                # transfer the path format from rgb-path to lmdb-key
                if file_path[0] == ".":
                    file_path = file_path.replace("./datasets\\", "")
                binary = txn.get(file_path.encode())
                landmark = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))
                landmark = self.rescale_landmarks(
                    np.float32(landmark),
                    original_size=256,
                    new_size=self.config["resolution"],
                )
        return landmark

    def to_tensor(self, img):
        """
        Convert an image to a PyTorch tensor.
        """
        return T.ToTensor()(img)

    def normalize(self, img):
        """
        Normalize an image.
        """
        mean = self.config["mean"]
        std = self.config["std"]
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        # Create a dictionary of arguments
        kwargs = {"image": img}

        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs["keypoints"] = landmark
            kwargs["keypoint_params"] = A.KeypointParams(format="xy")
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs["mask"] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)

        # Get the augmented image, landmark, and mask
        augmented_img = transformed["image"]
        augmented_landmark = transformed.get("keypoints")
        augmented_mask = transformed.get("mask", mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def __getitem__(self, index, no_norm=False):
        """
        Returns the data point at the given index.

        Args:
            index (int): The index of the data point.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Get the image paths and label
        image_paths = self.data_dict["image"][index]
        label = self.data_dict["label"][index]

        if not isinstance(image_paths, list):
            image_paths = [
                image_paths
            ]  # for the image-level IO, only one frame is used

        image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        augmentation_seed = None

        for image_path in image_paths:
            # Initialize a new seed for data augmentation at the start of each video
            if self.video_level and image_path == image_paths[0]:
                augmentation_seed = random.randint(0, 2**32 - 1)

            # Get the mask and landmark paths
            mask_path = image_path.replace("frames", "masks")  # Use .png for mask
            landmark_path = image_path.replace("frames", "landmarks").replace(
                ".png", ".npy"
            )  # Use .npy for landmark

            # Load the image
            try:
                image = self.load_rgb(image_path)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load image at dataset index {index}: {image_path}"
                ) from e
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark (if needed)
            
            if self.config["with_mask"]:
                mask = self.load_mask(mask_path)
            else:
                mask = None
            if self.config["with_landmark"] and self.mode=="train":
                landmarks = self.load_landmark(landmark_path)
            else:
                landmarks = None

            # Do Data Augmentation
            if self.mode == "train" and self.config["use_data_augmentation"]:
                image_trans, landmarks_trans, mask_trans = self.data_aug(
                    image, landmarks, mask, augmentation_seed
                )
            else:
                image_trans, landmarks_trans, mask_trans = (
                    deepcopy(image),
                    deepcopy(landmarks),
                    deepcopy(mask),
                )

            # To tensor and normalize
            if not no_norm:
                image_trans = self.normalize(self.to_tensor(image_trans))
                image_tensors.append(image_trans)
                if(self.mode=="train"):
                    if self.config["with_landmark"]:
                        landmarks_trans = torch.from_numpy(landmarks)
                    if self.config["with_mask"]:
                        mask_trans = torch.from_numpy(mask_trans)

                landmark_tensors.append(landmarks_trans)
                mask_tensors.append(mask_trans)

        if self.video_level:
            # Stack image tensors along a new dimension (time)
            image_tensors = torch.stack(image_tensors, dim=0)
            # Stack landmark and mask tensors along a new dimension (time)
            if not any(
                landmark is None or (isinstance(landmark, list) and None in landmark)
                for landmark in landmark_tensors
            ):
                landmark_tensors = torch.stack(landmark_tensors, dim=0)
            if not any(
                m is None or (isinstance(m, list) and None in m) for m in mask_tensors
            ):
                mask_tensors = torch.stack(mask_tensors, dim=0)
        else:
            # Get the first image tensor
            image_tensors = image_tensors[0]
            # Get the first landmark and mask tensors
            if not any(
                landmark is None or (isinstance(landmark, list) and None in landmark)
                for landmark in landmark_tensors
            ):
                landmark_tensors = landmark_tensors[0]
            if not any(
                m is None or (isinstance(m, list) and None in m) for m in mask_tensors
            ):
                mask_tensors = mask_tensors[0]

        return image_tensors, label, landmark_tensors, mask_tensors

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        images, labels, landmarks, masks = zip(*batch)

        # Stack the image, label, landmark, and mask tensors
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        # Special case for landmarks and masks if they are None
        if not any(
            landmark is None or (isinstance(landmark, list) and None in landmark)
            for landmark in landmarks
        ):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if not any(m is None or (isinstance(m, list) and None in m) for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict["image"] = images
        data_dict["label"] = labels
        data_dict["landmark"] = landmarks
        data_dict["mask"] = masks

        return data_dict

    def __len__(self):
        assert len(self.image_list) == len(
            self.label_list
        ), "Number of images and labels are not equal"
        return len(self.image_list)


class CombinedTrainDataset(data.Dataset):
    """Concatenate independent train datasets, each with its own LMDB."""

    def __init__(self, datasets):
        if not datasets:
            raise ValueError("CombinedTrainDataset requires at least one dataset")
        self.datasets = datasets
        self.offsets = []
        self.data_dict = {"image": [], "label": []}
        total = 0
        for dataset in datasets:
            self.offsets.append(total)
            total += len(dataset)
            self.data_dict["image"].extend(dataset.data_dict["image"])
            self.data_dict["label"].extend(dataset.data_dict["label"])
        self.total_length = total

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        for dataset_index in range(len(self.datasets) - 1, -1, -1):
            if index >= self.offsets[dataset_index]:
                return self.datasets[dataset_index][
                    index - self.offsets[dataset_index]
                ]
        raise IndexError(index)

    @staticmethod
    def collate_fn(batch):
        return DeepfakeAbstractBaseDataset.collate_fn(batch)


if __name__ == "__main__":
    with open("./training/config/detector/video_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_set = DeepfakeAbstractBaseDataset(
        config=config,
        mode="train",
    )
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config["train_batchSize"],
        shuffle=True,
        num_workers=0,
        collate_fn=train_set.collate_fn,
    )
    from tqdm import tqdm

    for iteration, batch in enumerate(tqdm(train_data_loader)):
        # print(iteration)
        ...
        # if iteration > 10:
        #     break
