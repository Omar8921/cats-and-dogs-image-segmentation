import torch
from deep_learning.utils import validate_batch
from torch.utils.data import DataLoader

def evaluate_model(test_ldr: DataLoader, device: torch.device,
                   model, criterion):
    total_loss = 0.0
    total_miou = 0.0

    for images, masks in test_ldr:
        images = images.to(device)
        masks = masks.to(device)
        loss, miou = validate_batch(model, criterion, images, masks)
        total_loss += loss
        total_miou += miou

    test_loss = total_loss / len(test_ldr)
    test_miou = total_miou / len(test_ldr)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test MIOU: {test_miou:.4f}')

    return test_loss, test_miou