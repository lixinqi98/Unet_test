import argparse
import os
import glob
from PIL import Image
import numpy as np
import torch
import SimpleITK as sitk
from utils.dice_score import multiclass_dice_coeff, dice_coeff

# import torch.nn.functional as F

# from torchvision import transforms

# from utils.data_loading import BasicDataset
# from unet import UNet
# from utils.utils import plot_img_and_mask
def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='./checkpoint_epoch500.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input_folder', '-i', default='./data/test/img')
    parser.add_argument('--output_folder', '-o', default='./data/output')

    args = parser.parse_args()
                    
    image_list = []
    dice_score = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    i = 0
    for filename in glob.glob(os.path.join(args.input_folder, '*.npy')): #assuming nii
        # i += 1
        # if i>=5:
        #     break
        inputpath = filename
        outputpath = os.path.join(args.output_folder, filename.split('/')[-1])

        print(f"python predict.py -m {args.model} -i {inputpath} -o {outputpath} --scale 1")

        os.system(f"python predict.py -m {args.model} -i {inputpath} -o {outputpath} --scale 1")

        predict = torch.from_numpy(np.load(outputpath))
        mask = torch.from_numpy(np.load(os.path.join('./test_data/masks', filename.split('/')[-1])))
        pred = predict.to(device=device, dtype=torch.long)
        orig = mask.to(device=device, dtype=torch.long)
        pred_np = np.load(outputpath)
        # orig = np.load(os.path.join('./data/test/masks', filename.split('/')[-1]))
        pred_np = sitk.GetImageFromArray(pred_np)
        sitk.WriteImage(pred_np, os.path.join('./test_data/output_nii', filename.split('/')[-1] + '.nii'))

        # score = dice(pred, orig)
        score = dice_coeff(pred, orig)
        dice_score.append(score)
        print(f"{filename} - {score}")
    print(np.array(dice_score))
    print(np.mean(np.array(dice_score)))


        