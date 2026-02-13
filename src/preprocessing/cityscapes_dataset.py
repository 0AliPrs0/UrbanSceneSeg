import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from src.preprocessing.mask_utils import process_mask_to_train_id


class CityscapesDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_names, mask_names, resize_h=96, resize_w=256, augment=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = img_names
        self.mask_names = mask_names
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.augment = augment
        self.to_tensor = transforms.ToTensor()
        # Normalization parameters from standard ImageNet pre-training, common practice.
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        # Image Loading and Preprocessing
        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.resize_w, self.resize_h), Image.BILINEAR)
        img_t = self.to_tensor(img)            # [3, H, W]
        img_t = self.normalize(img_t)

        # Mask Loading and Preprocessing
        mask_t = torch.from_numpy(process_mask_to_train_id(mask_path, self.resize_h, self.resize_w)).long()  # [H, W], int64

        if self.augment:
            # Augmentation uses F from torchvision.transforms (as F in your original code)
            # Random horizontal flip
            if random.random() < 0.5:
                img_t = torch.flip(img_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[1])
            # Random rotation (-15 to +15 degrees)
            angle = random.uniform(-15, 15)
            # Rotate image (Bilinear interpolation for image)
            img_t = F.rotate(img_t, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            # Rotate mask (Nearest neighbor for mask)
            mask_t = F.rotate(mask_t.unsqueeze(0), angle,
                              interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)
            # Color jitter (only on image)
            jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            img_t = jitter(img_t)

        return img_t, mask_t
