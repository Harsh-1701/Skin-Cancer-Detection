"""
Image preprocessing and augmentation
for EfficientNet-B4
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_SIZE = 380


# -------------------------------------------------------
# TRAIN TRANSFORM
# -------------------------------------------------------

train_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE,
    ),

    A.HorizontalFlip(
        p=0.5,
    ),

    A.VerticalFlip(
        p=0.5,
    ),

    A.RandomRotate90(
        p=0.5,
    ),

    A.Affine(
        translate_percent=0.05,
        scale=(0.90, 1.10),
        rotate=(-20, 20),
        border_mode=0,
        p=0.5,
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5,
    ),

    A.GaussianBlur(
        blur_limit=(3, 5),
        p=0.15,
    ),

    A.Normalize(),

    ToTensorV2(),
])


# -------------------------------------------------------
# VALIDATION / TEST TRANSFORM
# -------------------------------------------------------

val_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE,
    ),

    A.Normalize(),

    ToTensorV2(),
])