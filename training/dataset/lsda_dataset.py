import sys
sys.path.append('.')

import os
import sys
import json
import math
import yaml

import numpy as np
import cv2
import random
from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms as T


import skimage.draw
import albumentations as alb
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur, RandomResizedCrop
from torch.utils.data.sampler import Sampler
from .abstract_dataset import DeepfakeAbstractBaseDataset


private_path_prefix = './training'

fake_dict = {
    'real': 0,
    'Deepfakes': 1, 
    'Face2Face': 2,
    'FaceSwap': 3, 
    'NeuralTextures': 4, 
    # 'Deepfakes_Face2Face': 5, 
    # 'Deepfakes_FaceSwap': 6, 
    # 'Deepfakes_NeuralTextures': 7, 
    # 'Deepfakes_real': 8, 
    # 'Face2Face_FaceSwap': 9, 
    # 'Face2Face_NeuralTextures': 10, 
    # 'Face2Face_real': 11, 
    # 'FaceSwap_NeuralTextures': 12, 
    # 'FaceSwap_real': 13, 
    # 'NeuralTextures_real': 14,
}



class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
	def apply(self,img,**params):
		return self.randomdownscale(img)

	def randomdownscale(self,img):
		keep_ratio=True
		keep_input_shape=True
		H,W,C=img.shape
		ratio_list=[2,4]
		r=ratio_list[np.random.randint(len(ratio_list))]
		img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
		if keep_input_shape:
			img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

		return img_ds


augmentation_methods = alb.Compose([
    # alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=0.5),
    # HorizontalFlip(p=0.5),
    # RandomDownScale(p=0.5),
    # alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
    GaussianBlur(blur_limit=[3, 7], p=0.5)
], p=1.)

augmentation_methods2 = alb.Compose([
    alb.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1,0.1), p=0.5),
    HorizontalFlip(p=0.5),
    RandomDownScale(p=0.5),
    alb.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
    alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.5),
],
additional_targets={f'image1':'image', f'image2':'image', f'image3':'image', f'image4':'image'},
p=1.)

normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                     std =[0.5, 0.5, 0.5])
transforms1 = T.Compose([
            T.ToTensor(),
            normalize
        ])

#==========================================

def load_rgb(file_path, size=256):
    assert os.path.exists(file_path), f"{file_path} is not exists"
    img = cv2.imread(file_path)
    if img is None: 
        raise ValueError('Img is None: {}'.format(file_path))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(np.array(img, dtype=np.uint8))


def load_mask(file_path, size=256):
    mask = cv2.imread(file_path, 0)
    if mask is None:
        mask = np.zeros((size, size))

    mask = cv2.resize(mask, (size, size))/255
    mask = np.expand_dims(mask, axis=2)
    return np.float32(mask)


def add_gaussian_noise(ins, mean=0, stddev=0.1):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return torch.clamp(ins + noise, -1, 1)


# class RandomBlur(object):
#     """ Randomly blur an image 
#     """
#     def __init__(self, ratio,)

# class RandomCompression(object):
#     """ Randomly compress an image 
#     """

class CustomSampler(Sampler):
    def __init__(self, num_groups=2*360, n_frame_per_vid=32, videos_per_group=5, batch_size=10):
        self.num_groups = num_groups
        self.n_frame_per_vid = n_frame_per_vid
        self.videos_per_group = videos_per_group
        self.batch_size = batch_size
        assert self.batch_size % self.videos_per_group == 0, "Batch size should be a multiple of videos_per_group."
        self.groups_per_batch = self.batch_size // self.videos_per_group

    def __iter__(self):
        group_indices = list(range(self.num_groups))
        random.shuffle(group_indices)

        # For each batch
        for i in range(0, len(group_indices), self.groups_per_batch):
            selected_groups = group_indices[i:i+self.groups_per_batch]
            
            # For each group
            for group in selected_groups:
                frame_idx = random.randint(0, self.n_frame_per_vid - 1)  # Random frame index for this group's videos
                
                # Return the frame for each video in this group using the same frame_idx
                for video_offset in range(self.videos_per_group):
                    yield group * self.videos_per_group * self.n_frame_per_vid + video_offset * self.n_frame_per_vid + frame_idx

    def __len__(self):
        return self.num_groups * self.videos_per_group  # Total frames



class LSDADataset(DeepfakeAbstractBaseDataset):

    on_3060 = "3060" in torch.cuda.get_device_name()
    transfer_dict = {
        'youtube':'FF-real',
        'Deepfakes':'FF-DF',
        'Face2Face':'FF-F2F',
        'FaceSwap':'FF-FS',
        'NeuralTextures':'FF-NT'


    }
    if on_3060:
        data_root = r'F:\Datasets\rgb\FaceForensics++'
    else:
        data_root = r'./datasets/FaceForensics++'
    data_list = {
        'test': r'./datasets/rgb/FaceForensics++/test.json',
        'train': r'./datasets/rgb/FaceForensics++/train.json',
        'eval': r'./datasets/rgb/FaceForensics++/val.json'  
    }

    def __init__(self, config=None, mode='train', with_dataset=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']):
        super().__init__(config, mode)
        self.mode = mode
        self.res = config['resolution']
        self.fake_dict = fake_dict
        # transform
        self.normalize = T.Normalize(mean=config['mean'],
                                     std =config['std'])
        # data aug and transform
        self.transforms1 = T.Compose([
            T.ToTensor(),
            self.normalize
        ])
        self.img_lines = []
        self.config=config
        self.videos_per_group = 1 + len(with_dataset)
        self._warned_videos = set()
        with open(self.config['dataset_json_folder']+'/FaceForensics++.json', 'r') as fd:
            self.img_json = json.load(fd)
        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            img_lines = []
            skipped_groups = []
            for pair in data:
                r1, r2 = pair
                for source, target in ((r1, r2), (r2, r1)):
                    group = [('youtube', source, 0)] + [
                        (fake_d, '{}_{}'.format(source, target), self.fake_dict[fake_d])
                        for fake_d in with_dataset
                    ]
                    if not self._group_has_enough_frames(group):
                        skipped_groups.append((source, target))
                        continue
                    for instance_type, video_name, label in group:
                        name = '{}/{}'.format(instance_type, video_name)
                        for frame_idx in range(config['frame_num'][mode]):
                            img_lines.append((name, frame_idx, label, mode))

        self.num_groups = len(img_lines) // (
            self.videos_per_group * config['frame_num'][mode]
        )
        assert self.num_groups > 0, "No complete LSDA training groups were found."
        if skipped_groups:
            print(
                "LSDA dataset: skipped {} incomplete groups; using {} groups.".format(
                    len(skipped_groups), self.num_groups
                )
            )
        self.img_lines.extend(img_lines)

    def _group_has_enough_frames(self, group):
        required_frames = self.config['frame_num'][self.mode]
        dataset = self.img_json[self.data_root.split(os.path.sep)[-1]]
        for instance_type, video_name, _ in group:
            domain = self.transfer_dict[instance_type]
            video = dataset[domain][self.mode][self.compression].get(video_name)
            if video is None or len(video.get('frames', [])) < required_frames:
                return False
        return True


    def get_ids_from_path(self, path):
        parts = path.split('/')
        try:
            if 'youtube' in path:
                return [int(parts[-1])]
            else:
                return list(map(int, parts[-1].split('_')))
        except:      
            raise ValueError("wrong path: {}".format(path))

    def load_image(self, name, idx):
        instance_type, video_name = name.split('/')
        #其实并没有完全对应，而只是保证在同一video的目标时间区间内的一帧
        all_frames = self.img_json[self.data_root.split(os.path.sep)[-1]][self.transfer_dict[instance_type]]['train']['c23'][video_name]['frames']
        img_path = all_frames[idx]

        impath = img_path
        img = self.load_rgb(impath)
        return img

    def __getitem__(self, index):
        name, idx, label, mode = self.img_lines[index] #这个sampler的目的是不要取重复video的图。
        label = int(label)  # specific fake label from 1-4

        try:
            img = self.load_image(name, idx)
        except Exception as original_error:
            img, name, idx, label = self._load_fallback(index, original_error)

            
        if self.mode=='train':
            # do augmentation
            img = np.asarray(img) # convert PIL to numpy

            img = augmentation_methods2(image=img)['image']
            img = Image.fromarray(np.array(img, dtype=np.uint8)) # covnert numpy to PIL

            # transform with PIL as input
            img = self.transforms1(img)
        else:
            raise ValueError("Not implemented yet")

        return (img, label)

    def _load_fallback(self, index, original_error):
        frame_count = self.config['frame_num'][self.mode]
        group_stride = self.videos_per_group * frame_count
        video_offset = (index % group_stride) // frame_count

        candidates = []
        video_start = index - (index % frame_count)
        other_frames = [i for i in range(frame_count) if i != index % frame_count]
        random.shuffle(other_frames)
        candidates.extend(video_start + frame_idx for frame_idx in other_frames)

        # Keep the same domain/label if the entire source video is unreadable.
        other_groups = list(range(self.num_groups))
        random.shuffle(other_groups)
        current_group = index // group_stride
        for group_idx in other_groups:
            if group_idx == current_group:
                continue
            frame_idx = random.randrange(frame_count)
            candidates.append(
                group_idx * group_stride + video_offset * frame_count + frame_idx
            )
            if len(candidates) >= frame_count + 16:
                break

        last_error = original_error
        original_name = self.img_lines[index][0]
        for candidate_index in candidates:
            candidate_name, candidate_frame, candidate_label, _ = self.img_lines[candidate_index]
            try:
                image = self.load_image(candidate_name, candidate_frame)
                if original_name not in self._warned_videos:
                    print(
                        "LSDA dataset: unreadable sample {} frame {}: {}; "
                        "using {} frame {} instead.".format(
                            original_name,
                            self.img_lines[index][1],
                            original_error,
                            candidate_name,
                            candidate_frame,
                        )
                    )
                    self._warned_videos.add(original_name)
                return image, candidate_name, candidate_frame, int(candidate_label)
            except Exception as error:
                last_error = error

        raise RuntimeError(
            "Unable to load LSDA sample {} after {} bounded fallback attempts. "
            "Original error: {}; last error: {}".format(
                self.img_lines[index], len(candidates), original_error, last_error
            )
        ) from original_error



    def __len__(self):
        return len(self.img_lines)



    @staticmethod
    def collate_fn(batch):
        # Unzip the batch into images and labels
        images, labels = zip(*batch)

        # images, labels = zip(batch['image'], batch['label'])

        # image_list = []

        # for i in range(len(images)//5):
            
        #     img = images[i*5:(i+1)*5]

        #     # do augmentation
        #     imgs_aug = augmentation_methods2(image=np.asarray(img[0]), image1=np.asarray(img[1]), image2=np.asarray(img[2]), image3=np.asarray(img[3]), image4=np.asarray(img[4]))
        #     for k in imgs_aug:

        #         img_aug = Image.fromarray(np.array(imgs_aug[k], dtype=np.uint8)) # covnert numpy to PIL

        #     # transform with PIL as input
        #         img_aug = transforms1(img_aug)
        #         image_list.append(img_aug)

        # Stack the images and labels
        images = torch.stack(images, dim=0)  # Shape: (batch_size, c, h, w)
        labels = torch.tensor(labels, dtype=torch.long)

        bs, c, h, w = images.shape

        # Assume videos_per_group is 5 in our case
        videos_per_group = 5
        num_groups = bs // videos_per_group

        # Reshape to get the group dimension: (num_groups, videos_per_group, c, h, w)
        images_grouped = images.view(num_groups, videos_per_group, c, h, w)
        labels_grouped = labels.view(num_groups, videos_per_group)

        valid_indices = []
        for i, group in enumerate(labels_grouped):
            if set(group.numpy().tolist()) == {0, 1, 2, 3, 4}:
                valid_indices.append(i)
            # elif set(group.numpy().tolist()) == {0, 1, 2, 3}:
            #     valid_indices.append(i)
            # elif set(group.numpy().tolist()) == {0, 1, 2, 3, 4, 5}:
            #     valid_indices.append(i)
        
        images_grouped = images_grouped[valid_indices]
        labels_grouped = labels_grouped[valid_indices]

        if not valid_indices:
            raise ValueError("No valid groups found in this batch.")

        # # Shuffle the video order within each group
        # for i in range(num_groups):
        #     perm = torch.randperm(videos_per_group)
        #     images_grouped[i] = images_grouped[i, perm]
        #     labels_grouped[i] = labels_grouped[i, perm]

        # # Flatten back to original shape but with shuffled video order
        # images_shuffled = images_grouped.view(num_groups, videos_per_group, c, h, w)
        # labels_shuffled = labels_grouped.view(bs)

        return {'image': images_grouped, 'label': labels_grouped, 'mask': None, 'landmark': None}


if __name__ == '__main__':
    with open('./training/config/detector/lsda.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = LSDADataset(config=config, mode='train')
    custom_sampler = CustomSampler(num_groups=2*360, n_frame_per_vid=config['frame_num']['train'], batch_size=config['train_batchSize'], videos_per_group=5)
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            num_workers=0,
            sampler=custom_sampler, 
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break
