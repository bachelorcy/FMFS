import os
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# 禁用 albumentations 的版本检查
os.environ["ALBUMENTATIONS_CHECK_VERSION"] = "0"

import albumentations as A
from albumentations.pytorch import ToTensorV2




class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # 默认参数
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        self.resize_size = (224, 224)  # 调整尺寸 (H, W)
        self.mode = 'train'  # 模式：'train' 或 'test'
        print('\n参数...')
        for k, v in self.kwargs.items():
            setattr(self, k, v)
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


def get_transforms(cfg):
    """获取数据增强和预处理的变换"""
    if cfg.mode == 'train':
        return A.Compose([
            A.Resize(height=cfg.resize_size[0], width=cfg.resize_size[1],
                     interpolation=cv2.INTER_LINEAR),

            A.OneOf([
                A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=1),
                A.RandomGamma(p=1),
                A.CLAHE(p=1)
            ], p=0.8),

            A.HorizontalFlip(p=0.5),

            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # shift_limit
                scale=(1 - 0.1, 1 + 0.1),  # scale_limit
                rotate=(-15, 15),  # rotate_limit
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),

            A.RandomCrop(height=cfg.resize_size[0] - 30, width=cfg.resize_size[1] - 30, p=0.8),

            A.Resize(height=cfg.resize_size[0], width=cfg.resize_size[1],
                     interpolation=cv2.INTER_LINEAR),

            A.Normalize(mean=cfg.mean.squeeze(), std=cfg.std.squeeze()),
            ToTensorV2()
        ], additional_targets={'mask': 'mask'})
    else:
        return A.Compose([
            A.Resize(height=cfg.resize_size[0], width=cfg.resize_size[1],
                     interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=cfg.mean.squeeze(), std=cfg.std.squeeze()),
            ToTensorV2()
        ])


class CamDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.transform = get_transforms(cfg)
        self.img_dir = cfg.img_dir
        self.mask_dir = cfg.mask_dir
        self.img_names = sorted(os.listdir(self.img_dir))

        # 验证数据路径
        assert len(self.img_names) > 0, f"No images found in {self.img_dir}"
        for img_name in self.img_names:
            mask_name = os.path.splitext(img_name)[0] + '.png'
            assert os.path.exists(os.path.join(self.mask_dir, mask_name)), f"Mask file missing: {mask_name}"

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        # 使用cv2保持数据类型一致性
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)

        # 二值化处理（兼容多进程）
        mask = (mask > 127.5).astype(np.float32)  # 直接得到 [0, 1] 范围

        # 应用变换
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        return image, mask.unsqueeze(0)


def collate_fn(batch):
    images, masks = zip(*batch)
    return torch.stack(images, 0), torch.stack(masks, 0)

