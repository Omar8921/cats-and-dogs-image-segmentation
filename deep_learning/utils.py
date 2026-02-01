import cv2
import numpy as np
from PIL import Image
import torch
from deep_learning.config import IGNORE_INDEX, NUM_CLASSES

def read_rgb_image(fpath):
    img = cv2.imread(fpath)
    if img is None:
        raise FileNotFoundError
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def read_mask(fpath):
    mask = np.array(Image.open(fpath))
    if len(mask.shape) == 3:
        mask = np.squeeze(mask, axis=-1)

    return mask


def calculate_weights(num_classes, ignore_index, dataset):
    num_classes = 3
    ignore_index = 255
    class_counts = np.zeros(num_classes, dtype=np.float64)

    for i in range(len(dataset)):
        img, mask = dataset[i]
        vals, cnts = np.unique(mask, return_counts=True)
        for v, c in zip(vals, cnts):
            if v != ignore_index:
                class_counts[v] += c

    freq = class_counts / class_counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float)

    return weights


def calculate_miou(_masks, masks):
    pred = torch.argmax(_masks, dim=1)

    pred = pred.reshape(-1)
    true = masks.reshape(-1)

    valid = true != IGNORE_INDEX
    pred = pred[valid]
    true = true[valid]

    ious = []
    for cls in range(NUM_CLASSES):
        pred_i = pred == cls
        true_i = true == cls

        intersection = torch.sum(pred_i & true_i).item()
        union = (torch.sum(pred_i).item() +
                 torch.sum(true_i).item() -
                 intersection)

        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)

    miou = np.nanmean(ious)

    return miou


def train_batch(model, optimizer,
                seg_criterion, images, masks):
    model.train()
    _masks = model(images)
    loss = seg_criterion(_masks, masks)
        
    miou = calculate_miou(_masks, masks)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), miou


@torch.no_grad()
def validate_batch(model, seg_criterion, images, masks):
    model.eval()
    _masks = model(images)
    loss = seg_criterion(_masks, masks)
    miou = calculate_miou(_masks, masks)

    return loss.item(), miou


colormap = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0)
}

def mask_to_rgb(mask, colormap):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colormap.items():
        rgb[mask == class_id] = color

    return rgb