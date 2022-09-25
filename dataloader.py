from typing import Tuple
import yaml
import os
import multiprocessing

import cv2
import torch
import numpy as np
import albumentations as A
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image


class VOCDataset(Dataset):
    def __init__(self, img_dir, transform=None, trans_params=None):
        self.img_dir = img_dir
        img_dir = img_dir.split("/")
        img_dir[-1] = "labels"
        self.label_dir = os.path.join(*img_dir)
        self.image_names = os.listdir(self.img_dir)
        self.transform = transform
        self.trans_params = trans_params

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, str]:
        label_path = os.path.join(self.label_dir, os.path.splitext(self.image_names[index])[0] + ".txt")  # /PASCAL_VOC/labels/000009.txt
        img_path = os.path.join(self.img_dir, self.image_names[index])  # /PASCAL_VOC/images/000009.jpg
        image = np.array(Image.open(img_path).convert("RGB"))  # albumentation을 적용하기 위해 np.array로 변환합니다.

        labels = None
        if os.path.exists(label_path):
            # np.roll: (class, cx, cy, w, h) -> (cx, cy, w, h, class)
            # np.loadtxt: txt 파일에서 data 불러오기
            labels = np.array(np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist())
            # labels = np.loadtxt(label_path).reshape(-1, 5)

        if self.transform:
            # apply albumentations
            augmentations = self.transform(image=image, bboxes=labels)
            image = augmentations['image']
            targets = augmentations['bboxes']

            # for DataLoader
            # lables: ndarray -> tensor
            # dimension: [batch, cx, cy, w, h, class]
            if targets is not None:
                targets = torch.zeros((len(labels), 6))
                targets[:, 1:] = torch.tensor(labels)
        else:
            targets = labels

        return image, targets, img_path


def collate_fn(batch) -> Tuple[Tensor, Tensor, Tensor]:
    imgs, targets, paths = list(zip(*batch))
    # 빈 박스 제거하기
    targets = [boxes for boxes in targets if boxes is not None]
    # index 설정하기
    for b_i, boxes in enumerate(targets):
        boxes[:, 0] = b_i
    targets = torch.cat(targets, 0)
    imgs = torch.stack([img for img in imgs])
    return imgs, targets, paths


def get_transform(image_size: int, scale: float) -> Tuple[A.Compose, A.Compose]:
    train_transforms = A.Compose([
        # 이미지의 maxsize를 max_size로 rescale합니다. aspect ratio는 유지합니다.
        A.LongestMaxSize(max_size=int(image_size * scale)),
        # min_size보다 작으면 pad
        A.PadIfNeeded(min_height=int(image_size * scale), min_width=int(image_size * scale),
                      border_mode=cv2.BORDER_CONSTANT),
        # random crop
        A.RandomCrop(width=image_size, height=image_size),
        # brightness, contrast, saturation을 무작위로 변경합니다.
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        # transforms 중 하나를 선택해 적용합니다.
        A.OneOf([
            # shift, scale, rotate 를 무작위로 적용합니다.
            A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            # affine 변환
            A.Affine(shear=15, p=0.5)
        ], p=1.0),
        # 수평 뒤집기
        A.HorizontalFlip(p=0.5),
        # blur
        A.Blur(p=0.1),
        # Contrast Limited Adaptive Histogram Equalization 적용
        A.CLAHE(p=0.1),
        # 각 채널의 bit 감소
        A.Posterize(p=0.1),
        # grayscale로 변환
        A.ToGray(p=0.1),
        # 무작위로 channel을 섞기
        A.ChannelShuffle(p=0.05),
        # normalize
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2()
    ],
        # (x1, y1, x2, y2) -> (cx, cy, w, h)
        bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
    )

    # for validation
    val_transforms = A.Compose([
        A.LongestMaxSize(max_size=int(image_size * scale)),
        A.PadIfNeeded(min_height=int(image_size * scale), min_width=int(image_size * scale),
                      border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(),
    ],
        bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
    )

    return train_transforms, val_transforms


def get_loader(data_file: str, image_size: int, batch_size: int, scale: float) -> Tuple[DataLoader, DataLoader]:
    with open(data_file) as f:
        data_file = yaml.load(f, Loader=yaml.FullLoader)

    train_dir = os.path.join(data_file["path"], data_file["train"])
    val_dir = os.path.join(data_file["path"], data_file["val"])

    train_ds = VOCDataset(train_dir)
    val_ds = VOCDataset(val_dir)

    train_ds.transform, val_ds.transform = get_transform(image_size, scale)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=multiprocessing.cpu_count() - 1)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=multiprocessing.cpu_count() - 1)

    return train_dl, val_dl


if __name__ == "__main__":
    train_dl, val_dl = get_loader("datasets/custom.yaml", 416, 2, 1)

    image, target, path = next(iter(val_dl))
