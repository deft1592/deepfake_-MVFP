"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
"""

import sys
from PIL import Image
sys.path.append(".")
from training.dataset.lightSimulationModule import  DECALightEditor
import cv2
import yaml
import torch,random
import numpy as np
from copy import deepcopy
import albumentations as A
from training.dataset.albu import IsotropicResize
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.sbi_api_real import SBI_API
import os
import json



# clsname2cls_path={
#     'Real': ['./datasets/rgb/FaceForensics++/original_sequences/actors/c23/frames','./datasets/rgb/FaceForensics++/original_sequences/youtube/c23/frames'],
#     'DF': ['./datasets/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames'],
#     'F2F':['./datasets/rgb/FaceForensics++/manipulated_sequences/Face2Face/c23/frames'],
#     'FS':['./datasets/rgb/FaceForensics++/manipulated_sequences/FaceSwap/c23/frames'],
#     'NT':['./datasets/rgb/FaceForensics++/manipulated_sequences/NeuralTextures/c23/frames'],
# }

    
STR2LABEL = {
    'original_sequences': 0,
    'Deepfakes': 1,
    'Face2Face': 2,
    'FaceSwap': 3,
    'NeuralTextures': 4,
}
ff_fake_list=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']


class MainDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None,mode="train"):
        super().__init__(config, mode)
        
        """
        TSNE数据集类，从每个类别的文件夹中随机抽取指定数量的图片
        
        Args:
            label_dict: 标签字典，格式为 {label_id: 'class_name'}
            samples_per_class: 每个类别抽取的图片数量
            transform: 图像变换
        """
        self.label_dict ={
        0: 'Real', 1: 'DF', 2: 'F2F', 3: 'FS', 4: 'NT', 
    }
        
        # self.mode=mode
        # self.config=config
        self.samples_per_class =500
        # 数据增强概率参数
        self.aug_prob = 0.5  # 整体增强概率
        self.occlusion_probs = {
            #'half_face': 0.5,
            'landmark': 1,
        }
        self.editor=DECALightEditor()
        self.augmentation = A.Compose([
        A.RandomBrightnessContrast(p=0.8),
        A.HueSaturationValue(p=0.8),
        A.CLAHE(p=0.5),
        A.Blur(blur_limit=3, p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
    ])
        # Init SBI
        self.sbi = SBI_API(phase="train", image_size=self.config["resolution"])
        
        # Init data augmentation method
        self.transform = self.init_data_aug_method()

        self.samples = []
        self.real_samples=[]
        self.df_samples=[]
        self.f2f_samples=[]
        self.nt_samples=[]
        self.fs_samples=[]

        self._build_samples_from_frame_paths(self.image_list,self.label_list)

     
    def _build_samples_from_frame_paths(self, frame_paths,label_list):
        """
        1. 从所有 frame_paths 中随机采样 5000
        2. 根据路径字符串标注 label (0~4)
        3. 保存 real / fake 样本
        """
        print("Building dataset from frame paths...")


        real_frame_paths = [
            p for p, l in zip(frame_paths, label_list) if l == 0
        ]
        DF_frame_paths=[p for p in frame_paths if "Deepfakes" in p ]
        F2F_frame_paths=[p for p in frame_paths if "Face2Face" in p ]
        FS_frame_paths=[p for p in frame_paths if "FaceSwap" in p ]
        NT_frame_paths=[p for p in frame_paths if "NeuralTextures" in p ]
        real_frame_paths = random.sample(real_frame_paths, 500)
        DF_frame_paths=random.sample(DF_frame_paths, 500)
        F2F_frame_paths=random.sample(F2F_frame_paths, 500)
        FS_frame_paths=random.sample(FS_frame_paths, 500)
        NT_frame_paths=random.sample(NT_frame_paths, 500)


        self.real_samples = [(path, 0) for path in real_frame_paths]
        self.DF_samples = [(path, 1) for path in DF_frame_paths]
        self.F2F_samples = [(path, 2) for path in F2F_frame_paths]
        self.FS_samples = [(path, 3) for path in FS_frame_paths]
        self.NT_samples = [(path, 4) for path in NT_frame_paths]


    def  get_image_label_for_image(self,image_path,label):


        image = np.array(self.load_rgb(image_path))
        
        image=self.normalize(self.to_tensor(image))
        return image,label

    def __getitem__(self, index):
        # ===== real sample =====
        real_image_path, real_label = self.real_samples[index]
        DF_image_path,DF_label=self.DF_samples[index]
        F2F_image_path,F2F_label=self.F2F_samples[index]
        FS_image_path,FS_label=self.FS_samples[index]
        NT_image_path,NT_label=self.NT_samples[index]

        real_landmark_path = real_image_path.replace(
            "frames", "landmarks"
        ).replace(".png", ".npy")

        DF_image, DF_label = self.get_image_label_for_image(DF_image_path, 1)
        F2F_image, F2F_label = self.get_image_label_for_image(F2F_image_path, 2)
        FS_image, FS_label = self.get_image_label_for_image(FS_image_path, 3)
        NT_image, NT_label = self.get_image_label_for_image(NT_image_path, 4)

        real_landmark = self.load_landmark(real_landmark_path).astype(np.int32)
        real_image = np.array(self.load_rgb(real_image_path))

        # SBI 生成 fake
        fake_image, real_image = self.sbi(
            real_image, real_landmark
        )
        # fake_image, real_image = self.sbi(
        #     real_image, self.editor, real_landmark
        # )
        if fake_image is None:
            fake_image = deepcopy(real_image)
            fake_label = 0
            print("SBI生成失败，使用原图作为fake")
        else:
            fake_label = 5
        fake_image = self.apply_random_occlusion(fake_image, real_landmark)

        real_image = self.normalize(self.to_tensor(real_image))
        fake_image = self.normalize(self.to_tensor(fake_image))

     

        return {
            "real": (real_image, real_label),
            "DF": (DF_image, DF_label),
            "F2F": (F2F_image, F2F_label),
            "FS": (FS_image, FS_label),
            "NT": (NT_image, NT_label),
            "fake": (fake_image, fake_label),
        }



    @staticmethod
    def collate_fn(batch):
        """
        Collate function matching __getitem__ return format.
        """

        # unpack
        real_images, real_labels = zip(*[b["real"] for b in batch])
        DF_images, DF_labels     = zip(*[b["DF"]   for b in batch])
        F2F_images, F2F_labels   = zip(*[b["F2F"]  for b in batch])
        FS_images, FS_labels     = zip(*[b["FS"]   for b in batch])
        NT_images, NT_labels     = zip(*[b["NT"]   for b in batch])
        fake_images, fake_labels = zip(*[b["fake"] for b in batch])

        # stack images
        real_images = torch.stack(real_images, dim=0)
        DF_images   = torch.stack(DF_images, dim=0)
        F2F_images  = torch.stack(F2F_images, dim=0)
        FS_images   = torch.stack(FS_images, dim=0)
        NT_images   = torch.stack(NT_images, dim=0)
        fake_images = torch.stack(fake_images, dim=0)

        # labels
        real_labels = torch.LongTensor(real_labels)
        DF_labels   = torch.LongTensor(DF_labels)
        F2F_labels  = torch.LongTensor(F2F_labels)
        FS_labels   = torch.LongTensor(FS_labels)
        NT_labels   = torch.LongTensor(NT_labels)
        fake_labels = torch.LongTensor(fake_labels)

        # concatenate for training / evaluation
        images = torch.cat(
            [real_images, fake_images, DF_images, F2F_images, FS_images, NT_images],
            dim=0
        )
        labels = torch.cat(
            [real_labels, fake_labels, DF_labels, F2F_labels, FS_labels, NT_labels],
            dim=0
        )

        return {
            "image": images,
            "label": labels,
            "landmark": None,
            "mask": None,
        }

    def __len__(self):
        return len(self.real_samples)


    def apply_random_occlusion(self, image, landmarks):
        """随机应用三种遮挡增强"""
        if random.random() > self.aug_prob:
            return image
            
        # 随机选择增强类型
        aug_type = random.choices(
            population=list(self.occlusion_probs.keys()),
            weights=list(self.occlusion_probs.values()),
            k=1
        )[0]
        
        if aug_type == 'half_face':
            return self.half_face_occlusion(image)
        elif aug_type == 'landmark':
            return self.landmark_occlusion(image, landmarks)
        return image

    def half_face_occlusion(self, image):
        """水平/垂直遮挡半张脸"""
        h, w = image.shape[:2]
        if random.random() < 0.5:  # 水平遮挡
            if random.random() < 0.5:
                image[:, :w//2] = 0
            else:
                image[:, w//2:] = 0
        else:  # 垂直遮挡
            if random.random() < 0.5:
                image[:h//2, :] = 0
            else:
                image[h//2:, :] = 0
        return image

    def landmark_occlusion(self, image, landmarks):

        """基于landmark的凸包遮挡，50%概率遮挡，50%概率使用Albumentations增强区域"""
        # 定义可遮挡区域
        regions = {
            'left_eye': list(range(36, 42)),
            'right_eye': list(range(42, 48)),
            'nose': list(range(27, 36)),
            'mouth': list(range(48, 68)),
            'left_eyebrow': list(range(17, 22)),  
            'right_eyebrow': list(range(22, 27)), 
        }
        
        # 50%概率执行原遮挡
        for region_name, region_indices in regions.items():
            if random.random() < 0.5:
                points = np.array([landmarks[i] for i in region_indices], dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(image, hull, (0, 0, 0))
            
            # else:
            #     # 获取该区域的所有点
            #     points = np.array([landmarks[i] for i in region_indices], dtype=np.int32)
                
            #     # 创建区域的掩码
            #     mask = np.zeros(image.shape[:2], dtype=np.uint8)
            #     hull = cv2.convexHull(points)
            #     cv2.fillConvexPoly(mask, hull, 255)
                
            #     # 提取区域ROI
            #     x, y, w, h = cv2.boundingRect(hull)
            #     region_roi = image[y:y+h, x:x+w].copy()
            #     mask_roi = mask[y:y+h, x:x+w]
                
            #     # 应用Albumentations增强
            #     augmented = self.augmentation(image=region_roi, mask=mask_roi)
            #     augmented_roi = augmented['image']
                
            #     # 将增强后的ROI混合回原图
            #     # 使用掩码确保只替换目标区域
            #     image[y:y+h, x:x+w] = np.where(
            #         mask_roi[:, :, np.newaxis].astype(bool),
            #         augmented_roi,
            #         image[y:y+h, x:x+w]
            #     )
        
        return image

    def half_image_occlusion(self, image):
        """随机遮挡图像的一半区域"""
        h, w = image.shape[:2]
        directions = ['top', 'bottom', 'left', 'right']
        direction = random.choice(directions)
        
        if direction == 'top':
            image[:h//2, :] = 0
        elif direction == 'bottom':
            image[h//2:, :] = 0
        elif direction == 'left':
            image[:, :w//2] = 0
        else:
            image[:, w//2:] = 0
        return image
    # 添加清理方法
    def cleanup(self):
        """清理数据集相关资源"""
        if hasattr(self, 'editor') and self.editor is not None:
            self.editor.cleanup()
            self.editor = None
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        import gc
        gc.collect()

    def __del__(self):
        """析构函数，确保资源被清理"""
        self.cleanup()  
        

    def init_data_aug_method(self):
        trans = A.Compose(
            [
                A.HorizontalFlip(p=self.config["data_aug"]["flip_prob"]),
                A.Rotate(
                    limit=self.config["data_aug"]["rotate_limit"],
                    p=self.config["data_aug"]["rotate_prob"],
                ),
                A.GaussianBlur(
                    blur_limit=self.config["data_aug"]["blur_limit"],
                    p=self.config["data_aug"]["blur_prob"],
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
                            brightness_limit=self.config["data_aug"][
                                "brightness_limit"
                            ],
                            contrast_limit=self.config["data_aug"]["contrast_limit"],
                        ),
                        A.FancyPCA(),
                        A.HueSaturationValue(),
                    ],
                    p=0.5,
                ),
                A.ImageCompression(
                    quality_lower=self.config["data_aug"]["quality_lower"],
                    quality_upper=self.config["data_aug"]["quality_upper"],
                    p=0.5,
                ),
            ],
            additional_targets={"real": "sbi"},
        )
        return trans


if __name__ == "__main__":
    with open("./training/config/detector/deft.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_set = MainDataset(config=config, mode="train")
    
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=config["train_batchSize"],
        shuffle=True,
        num_workers=0,
        collate_fn=train_set.collate_fn,
    )
    
    from tqdm import tqdm

    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break
