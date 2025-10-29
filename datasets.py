import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torchvision.transforms.functional as F

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset


class CityScapes(Dataset):
    """
    CityScapes Datasets.

    Args:
        mode(str): "train", "val", "test"
        transforms: Argumentation and transforms
    """

    def __init__(self, mode, transforms=None):
        super().__init__()
        self.mode = mode
        self.transforms = transforms
        self.image_dir = Path("../../data/CityScapes/images") / self.mode
        self.mask_dir = Path("../../data/CityScapes/masks") / self.mode
        self.image_path = list(self.image_dir.rglob("*.png"))
        self.mask_path = list(self.mask_dir.rglob("*.png"))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img = Image.open(self.image_path[idx]).convert("RGB")
        msk = Image.open(self.mask_path[idx])

        if self.transforms:
            img, msk = self.transforms(img, msk)

        return img, msk


class TrainTransforms:
    def __init__(self, crop_size=(512, 512), hflip_prob=0.5):
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

    def __call__(self, image, mask):
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        if torch.rand(1) < self.hflip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        image = self.color_jitter(image)

        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.287, 0.325, 0.284], std=[0.187, 0.190, 0.187]
        )
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


class ValTransforms:
    def __init__(self, crop_size=(512, 512)):
        self.crop_size = crop_size

    def __call__(self, image, mask):
        image = F.center_crop(image, self.crop_size)
        mask = F.center_crop(mask, self.crop_size)

        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.287, 0.325, 0.284], std=[0.187, 0.190, 0.187]
        )
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


class MultiTrainTransforms:
    def __init__(self, crop_size=(512, 512), hflip_prob=0.5, scale_range=(0.5, 2.0)):
        """
        Args:
            crop_size: 최종 crop 크기 (H, W)
            hflip_prob: 좌우 반전 확률
            scale_range: 랜덤 스케일 비율 범위 (ex: 0.5~2.0배)
        """
        self.crop_size = crop_size
        self.hflip_prob = hflip_prob
        self.scale_range = scale_range
        self.color_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)

    def __call__(self, image, mask):
        # 1. Random scaling
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        new_h = int(image.height * scale)
        new_w = int(image.width * scale)
        image = F.resize(
            image, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR
        )
        mask = F.resize(mask, (new_h, new_w), interpolation=F.InterpolationMode.NEAREST)

        # 2. Random crop
        i, j, h, w = T.RandomCrop.get_params(image, output_size=self.crop_size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)

        # 3. Random horizontal flip
        if torch.rand(1) < self.hflip_prob:
            image = F.hflip(image)
            mask = F.hflip(mask)

        # 4. Color jitter (이미지만)
        image = self.color_jitter(image)

        # 5. ToTensor & Normalize
        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.287, 0.325, 0.284], std=[0.187, 0.190, 0.187]
        )
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


class MultiValTransforms:
    def __init__(self, crop_size=(512, 512)):
        self.crop_size = crop_size

    def __call__(self, image, mask):
        # Validation은 단일 scale로 고정 (보통 원본 그대로 or center crop)
        image = F.center_crop(image, self.crop_size)
        mask = F.center_crop(mask, self.crop_size)

        image = F.to_tensor(image)
        image = F.normalize(
            image, mean=[0.287, 0.325, 0.284], std=[0.187, 0.190, 0.187]
        )
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return image, mask


if __name__ == "__main__":
    ts_train = TrainTransforms()
    ds_train = CityScapes(mode="train", transforms=ts_train)

    image, mask = ds_train[0]
    if ts_train is not None:
        image = torch.permute(image, (1, 2, 0))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.show()
