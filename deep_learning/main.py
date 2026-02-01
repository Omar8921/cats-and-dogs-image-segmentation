import pandas as pd
from deep_learning.preprocess import preprocess_data
from deep_learning.transforms import train_transform, val_transform
from deep_learning.train import train_model
from deep_learning.evaluate import evaluate_model
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images_dir, masks_dir = 'data/images', 'data/masks'

    train_df = pd.read_csv('data/train.csv')
    train_dataset, trn_ldr = preprocess_data(train_df, images_dir,
                                             masks_dir, train_transform,
                                             shuffle=True)
    
    
    val_df = pd.read_csv('data/val.csv')
    val_dataset, val_ldr = preprocess_data(val_df, images_dir,
                                             masks_dir, val_transform,
                                             shuffle=False)
    
    test_df = pd.read_csv('data/test.csv')
    test_dataset, test_ldr = preprocess_data(test_df, images_dir,
                                             masks_dir, val_transform,
                                             shuffle=False)
    
    print('Preprocessing done.')

    MODEL_PATH = 'deep_learning/model'
    model, criterion, history = train_model(train_dataset, val_dataset,
                                            trn_ldr, val_ldr, device, MODEL_PATH)

    print('Model training done.')

    test_loss, test_miou = evaluate_model(test_ldr, device, model, criterion)

    print('Model evaluation done.')

if __name__ == '__main__':
    main()