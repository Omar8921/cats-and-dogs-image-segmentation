from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_learning.utils import (calculate_weights,
                                 train_batch,
                                 validate_batch)
from torch.utils.data import DataLoader, Dataset
from deep_learning.config import (EPOCHS, LR, DECAY, PATIENCE,
                                  IGNORE_INDEX, NUM_CLASSES)

from numpy import inf
from tqdm import tqdm
import json

def conv_gn_relu(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
        nn.GroupNorm(32, out_ch),
        nn.ReLU(inplace=True),
    )


class DoubleConvGN(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            conv_gn_relu(in_ch, out_ch),
            conv_gn_relu(out_ch, out_ch),
            nn.Dropout2d(dropout)
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConvGN(out_ch + skip_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()

        backbone = models.resnet34(weights="DEFAULT" if pretrained else None)

        # Encoder
        self.enc1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )                   # 64
        self.enc2 = backbone.layer1  # 64
        self.enc3 = backbone.layer2  # 128
        self.enc4 = backbone.layer3  # 256
        self.enc5 = backbone.layer4  # 512

        # Decoder
        self.up4 = UpBlock(512, 256, 256, dropout=0.2)
        self.up3 = UpBlock(256, 128, 128, dropout=0.2)
        self.up2 = UpBlock(128, 64, 64, dropout=0.1)
        self.up1 = UpBlock(64, 64, 64, dropout=0.1)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]

        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        x  = self.enc5(s4)

        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        x = self.head(x)

        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return x


def train_model(train_dataset: Dataset, val_dataset: Dataset,
                trn_ldr: DataLoader, val_ldr: DataLoader, 
                device: torch.device, save_path: str):
    weights = calculate_weights(NUM_CLASSES, IGNORE_INDEX, train_dataset)
    model = UNetResNet34(num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=DECAY)
    criterion_seg = nn.CrossEntropyLoss(weight=weights.float().to(device), ignore_index=IGNORE_INDEX, label_smoothing=0.05)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.3,
        patience=4,
        threshold=0.005,
        threshold_mode="abs",
        min_lr=1e-5,
        verbose=True
    )

    history = {
        "train_loss": [],
        "train_miou": [],
        "val_loss": [],
        "val_miou": [],
    }

    best_val = inf
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):

        train_loss_sum = 0
        train_miou_sum = 0
        train_batches = 0

        train_pbar = tqdm(trn_ldr, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]", leave=False)

        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.long().to(device)

            loss, miou = train_batch(
                model, optimizer,
                criterion_seg,
                images, masks,
            )

            train_loss_sum += loss
            train_miou_sum += miou
            train_batches += 1

            train_pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "mIoU": f"{miou:.3f}"
            })

        train_loss = train_loss_sum / train_batches
        train_miou = train_miou_sum / train_batches


        val_loss_sum = 0
        val_miou_sum = 0
        val_batches = 0

        val_pbar = tqdm(val_ldr, desc=f"Epoch {epoch}/{EPOCHS} [VAL]", leave=False)

        for images, masks in val_pbar:
            images = images.to(device)
            masks = masks.long().to(device)

            loss, miou = validate_batch(
                model, criterion_seg,
                images, masks,
            )

            val_loss_sum += loss
            val_miou_sum += miou
            val_batches += 1

            val_pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "mIoU": f"{miou:.3f}"
            })

        val_loss = val_loss_sum / val_batches
        val_miou = val_miou_sum / val_batches
        
        history["train_loss"].append(train_loss)
        history["train_miou"].append(train_miou)
        history["val_loss"].append(val_loss)
        history["val_miou"].append(val_miou)

        scheduler.step(val_miou)

        print(
            f"[Epoch {epoch}/{EPOCHS}] "
            f"Train: loss={train_loss:.4f}, mIoU={train_miou:.4f} | "
            f"Val: loss={val_loss:.4f},  mIoU={val_miou:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path + '/model_weights.pth')
            with open(save_path + '/history.json', 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=4)
            
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print('Early Stopping.')
                break
    
    best_model = UNetResNet34().to(device)
    best_model.load_state_dict(torch.load(save_path + '/model_weights.pth'))

    return best_model, criterion_seg, history