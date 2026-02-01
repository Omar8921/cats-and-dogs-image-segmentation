import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

train_transform = A.Compose([
    A.Resize(256, 256),

    A.OneOf([
        A.HorizontalFlip(p=1.0),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            interpolation=cv2.INTER_NEAREST,
            p=1.0
        ),
    ], p=0.6),

    A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10
        ),
    ], p=0.4),

    A.GaussianBlur(blur_limit=3, p=0.1),

    A.CoarseDropout(
        max_holes=4,
        max_height=24,
        max_width=24,
        fill_value=0,
        mask_fill_value=0,
        p=0.2
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2()
], additional_targets={"mask": "mask"})


val_transform = A.Compose([
    A.Resize(
        256, 256,
        interpolation=cv2.INTER_NEAREST
    ),

    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),

    ToTensorV2(),
], additional_targets={"mask": "mask"})

