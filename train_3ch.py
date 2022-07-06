import argparse
import logging
import sys
from pathlib import Path
from matplotlib.transforms import Transform
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import monai
from monai.data import list_data_collate
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss, dice_coeff, multiclass_dice_coeff
from evaluate import evaluate
from unet import UNet
from sklearn.model_selection import KFold, GroupKFold
from dataload_3ch import Data

dir_img = '/Users/mona/Library/CloudStorage/OneDrive-Personal/Cedars-sinai/Unet/original_axial_3ch/imgs'
dir_mask = '/Users/mona/Library/CloudStorage/OneDrive-Personal/Cedars-sinai/Unet/original_axial_3ch/masks'
dir_checkpoint = Path('./checkpoints/test')
display_name = 'test'



def main(device, 
        epochs: int = 5, 
        batch_size: int = 128, 
        learning_rate: float = 0.001, 
        test_percent: float = 0.2, 
        img_scale: float = 1, 
        amp: bool = False,
        channels: int = 3, 
        classes: int = 2):
    # 1. Create dataset, load .npy files
    data_util = Data()
    data_images, data_labels = data_util.load(dir_img, dir_mask)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', name=display_name)
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  img_scale=img_scale, amp=amp))
    n_test = len(data_images) * test_percent
    n_train = len(data_images) - n_test   

    net = UNet(n_channels=channels, n_classes=classes, bilinear=True)
    net.to(device)
    # net.apply(reset_weights)

    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Testing size:    {n_test}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    # 
    kfold = KFold(n_splits=5, shuffle=True)
    # test_scores = []
    global_step = 0
    best_model = net
    best_score = 0
    val_scores= []
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data_images)):
        print(f'Fold {fold}')
        print('----------------------------------')
        test_images = list(map(data_images.__getitem__, test_ids))
        test_labels = list(map(data_labels.__getitem__, test_ids))

        train_images = list(map(data_images.__getitem__, train_ids))
        train_labels = list(map(data_labels.__getitem__, train_ids))

        test_dicts = [{'image': image_name, 'label': label_name}
                for image_name, label_name in zip(test_images, test_labels)]

        train_dicts = [{'image': image_name, 'label': label_name}
                    for image_name, label_name in zip(train_images, train_labels)]
                    
        check_val = monai.data.Dataset(data=test_dicts, transform=data_util.val_transforms)
        check_train = monai.data.Dataset(data=train_dicts, transform=data_util.train_transforms)
        train_loader = DataLoader(check_train, batch_size=batch_size, num_workers=4, collate_fn=list_data_collate, pin_memory=False, shuffle=True)
        val_loader = DataLoader(check_val, batch_size=1, num_workers=4, collate_fn=list_data_collate, pin_memory=False)
        
    # net.apply(reset_weights)

        n_train = len(train_loader)
        for epoch in range(epochs):
            # net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    # net.train()
                    images = batch['image']
                    true_masks = batch['label']

                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = np.squeeze(true_masks.to(device=device, dtype=torch.long))

                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = net(images)
                        loss = criterion(masks_pred, true_masks) \
                            + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                        F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                        multiclass=True)

                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                    experiment.log({
                        'fold': fold,
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                    pbar.set_postfix(**{'loss (batch)': loss.item()})

                    # Evaluation round
                    division_step = n_train // 2
                    if division_step > 0:
                        if global_step % division_step == 0:
                            histograms = {}
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                            val_score = evaluate(net, val_loader, device)
                            scheduler.step(val_score)

                            logging.info('Validation Dice score: {}'.format(val_score))
                            experiment.log({
                                'fold': fold,
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
        val_scores.append(val_score.cpu())
        if val_score > best_score:
            best_model = net
            torch.save(best_model.state_dict(), str(dir_checkpoint / 'best_model_fold.pth'))
        logging.info(f'Best Model with val score {best_score} has been saved')
        print(f"In fold {fold}, val score {val_score}")
    print(f"The average dice score is {np.mean(np.array(val_scores))}")
    return best_model


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.00001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


def reset_weights(self, m):
    for layer in m.children:
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    main(device, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

