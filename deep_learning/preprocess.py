from torch.utils.data import Dataset, DataLoader
from deep_learning.utils import read_mask, read_rgb_image
import os
import pandas as pd
import albumentations as A
from typing import Tuple

import warnings

warnings.filterwarnings(
    "ignore",
    message="ShiftScaleRotate is a special case of Affine transform.*",
)

warnings.filterwarnings(
    "ignore",
    message="Argument\\(s\\).*are not valid for transform CoarseDropout",
)

class SegmentationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, masks_dir: str, transform: A.Compose):
        self.df = df
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index, :]
        file= row['Image']
        img_path = os.path.join(self.images_dir, file + '.jpg')
        mask_path = os.path.join(self.masks_dir, file + '.png')
        img = read_rgb_image(img_path)
        mask = read_mask(mask_path)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img = aug['image']
            mask = aug['mask'].long()
        
        return img, mask

def preprocess_data(df: pd.DataFrame, images_dir: str,
                    masks_dir: str, transform: A.Compose,
                    shuffle: bool) -> Tuple[Dataset, DataLoader]:
    
    dataset = SegmentationDataset(df, images_dir, masks_dir, transform)
    dataloader = DataLoader(dataset, 8, shuffle=shuffle)
    
    return dataset, dataloader