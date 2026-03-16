import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# VOC 2012 — 21 classes (background + 20 object classes)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Distinct RGB color for each class — used in visualization
VOC_COLORMAP = [
    (0,   0,   0),    # background
    (128, 0,   0),    # aeroplane
    (0,   128, 0),    # bicycle
    (128, 128, 0),    # bird
    (0,   0,   128),  # boat
    (128, 0,   128),  # bottle
    (0,   128, 128),  # bus
    (128, 128, 128),  # car
    (64,  0,   0),    # cat
    (192, 0,   0),    # chair
    (64,  128, 0),    # cow
    (192, 128, 0),    # diningtable
    (64,  0,   128),  # dog
    (192, 0,   128),  # horse
    (64,  128, 128),  # motorbike
    (192, 128, 128),  # person
    (0,   64,  0),    # pottedplant
    (128, 64,  0),    # sheep
    (0,   192, 0),    # sofa
    (128, 192, 0),    # train
    (0,   64,  128),  # tvmonitor
]


def get_transforms(split: str, image_size: int = 512):
    if split == "train":
        return A.Compose([
            A.RandomScale(scale_limit=(-0.5, 0.5), p=1.0),  # scale 0.5x to 1.5x
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,          # constant padding
                fill=0,
                fill_mask=255           # pad mask with ignore_index
            ),
            A.RandomCrop(height=image_size, width=image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=0.5
            ),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])

class VOCSegmentationDataset(Dataset):
    """
    PASCAL VOC 2012 Segmentation Dataset.

    Masks are PNG files where each pixel value is a class index (0-20).
    Boundary pixels are labeled 255 — handled via ignore_index in loss.
    """

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        image_size: int = 512,
        ignore_index: int = 255
    ):
        super().__init__()
        self.root = root
        self.split = split
        self.image_size = image_size
        self.ignore_index = ignore_index
        self.transforms = get_transforms(split, image_size)

        # Paths
        voc_root = os.path.join(root, "VOCdevkit", "VOC2012")
        self.img_dir  = os.path.join(voc_root, "JPEGImages")
        self.mask_dir = os.path.join(voc_root, "SegmentationClass")

        # Read split file → list of image IDs
        split_file = os.path.join(
            voc_root, "ImageSets", "Segmentation", f"{split}.txt"
        )
        with open(split_file, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]

        print(f"VOCSegmentationDataset [{split}]: {len(self.ids)} samples")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]

        # Load image as RGB numpy array
        img_path  = os.path.join(self.img_dir,  f"{img_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{img_id}.png")

        image = np.array(Image.open(img_path).convert("RGB"))   # (H, W, 3)
        mask  = np.array(Image.open(mask_path))                  # (H, W) — class indices

        # Apply synchronized transforms
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]   # Tensor (3, H, W), float32
        mask  = augmented["mask"]    # Tensor (H, W), uint8 → we fix dtype below

        # Mask must be long (int64) for CrossEntropyLoss
        mask = mask.long()

        return image, mask

    def get_raw(self, idx: int):
        """Returns raw PIL image + mask without transforms. Used for visualization."""
        img_id = self.ids[idx]
        image = Image.open(os.path.join(self.img_dir,  f"{img_id}.jpg")).convert("RGB")
        mask  = Image.open(os.path.join(self.mask_dir, f"{img_id}.png"))
        return image, mask