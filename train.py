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
from utils.data_loading import BasicDataset
from utils.dice_score import dice_loss, dice_coeff, multiclass_dice_coeff
from evaluate import evaluate
from unet import UNet
from sklearn.model_selection import KFold, GroupKFold
from dataload import transformation

dir_img = Path('/Users/mona/codeWorkSpace/github_repo/DeepShim/code/python/Pytorch-UNet/data/imgs')
dir_mask = Path('/Users/mona/codeWorkSpace/github_repo/DeepShim/code/python/Pytorch-UNet/data/masks')
dir_checkpoint = Path('./checkpoints/test')
global_step = 0

def main(device, 
         epochs: int = 5, 
         batch_size: int = 128, 
         learning_rate: float = 0.001, 
         test_percent: float = 0.1, 
         img_scale: float = 1, 
         amp: bool = False,
         channels: int = 1, 
         classes: int = 2):

    # 1. Create dataset, load .npy files
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    n_train = len(dataset) - int(test_percent * len(dataset))
    n_test = len(dataset) - n_train

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  test_num=n_test, img_scale=img_scale, amp=amp))
    trans = transformation()
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

    # Start group kfold
    test_kfold = KFold(n_splits=int(1/test_percent), shuffle=True)
    test_scores = []
    for fold, (train_ids, test_ids) in enumerate(test_kfold.split(dataset)):
        print(f'Fold {fold}')
        print('----------------------------------')

        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=test_sampler)
        net = UNet(n_channels=channels, n_classes=classes, bilinear=True)
        net.to(device)
        # net.apply(reset_weights)
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss()
        
        try:
            net = train_net(net, experiment, fold, train_ids, dataset, device, epochs, batch_size, amp, optimizer, scheduler, grad_scaler, criterion, trans)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            sys.exit(0)
        # test_score = evaluate(net, test_loader, device)
        
        net.eval()
        num_val_batches = len(test_loader)
        dice_score = 0

        # iterate over the validation set
        for batch in tqdm(test_loader, total=num_val_batches, desc='Test round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

            with torch.no_grad():
                # predict the mask
                mask_pred = net(image)

                # convert to one-hot format
                if net.n_classes == 1:
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            
            experiment.log({
                'model': fold,
                'Test Dice': dice_score,
                'test_images': wandb.Image(image[0].cpu()),
                'test_masks': {
                    'true': wandb.Image(mask_true[0].float().cpu()),
                    'pred': wandb.Image(torch.softmax(mask_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                }
            })
        # Fixes a potential division by zero error
        if num_val_batches != 0:
            test_score = dice_score
        test_score = dice_score / num_val_batches

        logging.info('Test Dice score for fold {}: {}'.format(fold, test_score))
        test_scores.append(test_score.cpu())
    print(f"Mean of Test Dice score {np.array(test_scores).mean()} and std is {np.array(test_scores).std()}")


def train_net(net, experiment, model_index, train_sampler, dataset,
              device, epochs, batch_size, amp, 
              optimizer, scheduler, grad_scaler, criterion, trans):

    # Begin training
    kfold = KFold(n_splits=5, shuffle=True)
    best_model = net
    best_score = 0
    val_score = 0
    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_sampler)):
        print(f"----------Cross Validation Fold {fold} --------------")
        train_sampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_ids)

        train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler, transform=trans.train_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=val_sampler, transform=trans.val_transform)
        n_train = len(train_loader)
        for epoch in range(epochs):
            net.train()
            epoch_loss = 0
            with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                for batch in train_loader:
                    images = batch['image']
                    true_masks = batch['mask']

                    assert images.shape[1] == net.n_channels, \
                        f'Network has been defined with {net.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'

                    images = images.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

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
                    global global_step
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
                    division_step = (n_train // (10 * batch_size))
                    # division_step = 10
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
                                'model': model_index,
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

            if val_score > best_score:
                best_model = net

    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    torch.save(best_model.state_dict(), str(dir_checkpoint / 'best_model_fold_{}.pth'.format(model_index)))
    logging.info(f'Best Model with val score {best_score} in fold {model_index} has been saved')
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

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # net = UNet(n_channels=1, n_classes=2, bilinear=True)

    # logging.info(f'Network:\n'
    #              f'\t{n_channels} input channels\n'
    #              f'\t{net.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

