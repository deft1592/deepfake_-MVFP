"""
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
"""

import sys

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
from training.dataset.sbi_api_lsm import SBI_API

class DEFTDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode="train"):
        super().__init__(config, mode)

        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [
            (img, label)
            for img, label in zip(self.image_list, self.label_list)
            if label == 0
        ]
        self.fake_imglist = [
            (img, label)
            for img, label in zip(self.image_list, self.label_list)
            if label == 1
        ]
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
        self.sbi = SBI_API(phase=self.mode, image_size=self.config["resolution"])
        
        # Init data augmentation method
        self.transform = self.init_data_aug_method()

        
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
          
    def __getitem__(self, index):
        
        real_image_path, real_label = self.real_imglist[index]
        # Get the landmark paths for real images
        real_landmark_path = real_image_path.replace("frames", "landmarks").replace(
            ".png", ".npy"
        )
        real_landmark = self.load_landmark(real_landmark_path).astype(np.int32)
        
        # Load the real images
        real_image = self.load_rgb(real_image_path)
        real_image = np.array(real_image)  # Convert to numpy array

        # 使用beta分布生成额外blended样本
        #sample_size=1
        #alpha=0.5
        #beta=0.5
        #samples=np.random(alpha,beta,sample_size)
        #fake_image=fake_image*target+(1-target)*real_image
        
        fake_image ,real_image= self.sbi(real_image,self.editor,real_landmark)
        
        if fake_image is None:
            fake_image = deepcopy(real_image)
            fake_label = 0
        else:
            fake_label = 1


        #加入随机遮挡五官区域
        fake_image = self.apply_random_occlusion(fake_image, real_landmark)
        
        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image))
        real_image_trans = self.normalize(self.to_tensor(real_image))

        return {
            "fake": (fake_image_trans, fake_label),
            "real": (real_image_trans, real_label),
        }

    def __len__(self):
        return len(self.real_imglist)
    
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
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor and label tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images, fake_labels = zip(*[data["fake"] for data in batch])
        real_images, real_labels = zip(*[data["real"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images = torch.stack(fake_images, dim=0)
        fake_labels = torch.LongTensor(fake_labels)
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)

        data_dict = {
            "image": images, 
            "label": labels,
            "landmark": None,
            "mask": None,
        }
        return data_dict

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
    train_set = DEFTDataset(config=config, mode="train")
    
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
