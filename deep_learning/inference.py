from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_learning.config import NUM_CLASSES
from deep_learning.utils import mask_to_rgb, colormap
from deep_learning.transforms import val_transform
import numpy as np
import cv2


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


model = None
model_ready = False

def load_model(weights_path='deep_learning/model/model_weights.pth'):
    model = UNetResNet34(NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True))
    return model

def init_inference(weights_path):
    global model, model_ready
    model = load_model(weights_path)
    model.eval()
    model_ready = True


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def unnormalize(img):
    return img * IMAGENET_STD + IMAGENET_MEAN

def run_image_segmentation(img_bytes: bytes):
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image bytes")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    aug = val_transform(image=img)
    img_tensor = aug["image"]

    with torch.no_grad():
        logits = model(img_tensor.unsqueeze(0))
        mask = torch.argmax(logits, dim=1).float()

    mask = F.interpolate(
        mask.unsqueeze(1),
        size=(orig_h, orig_w),
        mode="nearest"
    ).squeeze(1)

    img_vis = unnormalize(img_tensor)
    img_vis = F.interpolate(
        img_vis.unsqueeze(0),
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    img_vis = img_vis.clamp(0, 1).cpu().numpy()
    img_vis = (img_vis * 255).astype(np.uint8)
    img_vis = img_vis.transpose(1, 2, 0)

    rgb_mask = mask_to_rgb(mask.squeeze(0), colormap)

    alpha = 0.5
    overlay = cv2.addWeighted(img_vis, alpha, rgb_mask, 1 - alpha, 0)

    return rgb_mask, overlay