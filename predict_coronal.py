import argparse
import logging
from multiprocessing.connection import wait
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import monai
from monai.data import list_data_collate
from utils.data_loading import BasicDataset
from torch.utils.data import DataLoader
from unet import UNet
from utils.dice_score import dice_loss, dice_coeff, multiclass_dice_coeff
from utils.utils import plot_img_and_mask
import SimpleITK as sitk
from scipy.signal import resample_poly
from dataload import Data
# from myshow3d import myshow, myshow3d
import cv2
def predict_img(net,
                img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    # img = torch.from_numpy(full_img)
    # img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mask_pred = net(img)

        if net.n_classes == 1:
            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
            # compute the Dice score
            # dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
        else:
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
            # compute the Dice score, ignoring background
            # dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

        
        return torch.softmax(mask_pred, dim=1).argmax(dim=1)[0].float().cpu()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0)/ mask.shape[0]).astype(np.uint8))


def resolution_match(result_array, data_util, coord1, coord2):
    result_array_np = np.transpose(result_array, (2, 1, 0))
    result_array_temp = sitk.GetImageFromArray(result_array_np)
    result_array_temp.SetSpacing((data_util.resolution) + (1, ))

    # save temporary label
    writer = sitk.ImageFileWriter()
    writer.SetFileName('temp_seg.nii')
    writer.Execute(result_array_temp)

    files = [{"image": 'temp_seg.nii'}]

    files_ds = monai.data.Dataset(data=files, transform=data_util.reverse_transforms)
    files_loader = DataLoader(files_ds, batch_size=1, num_workers=0)

    for files_data in files_loader:
        files_images = files_data["image"]

        res = files_images.squeeze().data.numpy()

    # result_array = np.rint(res)
    result_array = res
    os.remove('./temp_seg.nii')

    # empty_array = np.zeros(original_shape)
    # empty_array[coord1[0]:coord2[0],coord1[1]:coord2[1],coord1[2]:coord2[2]] = result_array

    itk_image = sitk.GetImageFromArray(result_array)
    return itk_image


if __name__ == '__main__':
    input_path = '/Users/mona/workSpace/github_repo/DeepShim/code/python/Pytorch-UNet/data/target_data_coronal_1ch_equal_resolutionmatch'
    path = '/Users/mona/workSpace/github_repo/DeepShim/code/python/Pytorch-UNet/dataoutput/target_data_coronal_1ch_equal_resolutionmatch'
    model = f"checkpoints/coronal_2/best_model_fold.pth"

    if not os.path.exists(path):
        os.makedirs(path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_util = Data()

    print(f'Using device {device}')

    net = UNet(n_channels=1, n_classes=2)
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    print('Model loaded!')


    input_files = glob.glob(f"{input_path}/*.nii")
    input_files.sort(key=lambda x:int((x.split("/")[-1]).split('_')[-1][:-4]))
    masks = []
    images = []
    for image in input_files:
        original_shape, crop_shape, coord1, coord2, resampled_size, original_resolution = data_util.statistics_crop(image, data_util.resolution)
        files = [{"image": image}]
        test_ds = monai.data.Dataset(data=files, transform=data_util.test_transforms)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate, pin_memory=False)
        for test_data in test_loader:
            test_image = test_data["image"].to(device)
            mask_slice = predict_img(net=net,
                            img=test_image,
                            scale_factor=1,
                            out_threshold=0.5,
                            device=device)
        result_array = mask_slice.squeeze().data.cpu().numpy()
        result_array = result_array[0:resampled_size[0],0:resampled_size[1]]
        
        result_seg = data_util.from_numpy_to_itk(result_array, image)
        result = os.path.join(path, image.split("/")[-1])
        writer = sitk.ImageFileWriter()
        writer.SetFileName(result)
        writer.Execute(result_seg)

        masks.append(result_array)
        images.append((test_image[0].squeeze().data.cpu().numpy())[0:resampled_size[0],0:resampled_size[1]])

    masks = np.dstack(masks)
    images = np.dstack(images)

    masks = masks.transpose((0, 2, 1))
    images = masks.transpose((0, 2, 1))

    # masks_img = data_util.from_numpy_to_itk(masks, image)
    # images_img = data_util.from_numpy_to_itk(images, image)

    if not (original_resolution == data_util.resolution):
        data_util.init_reverse(crop_shape=(crop_shape+(len(input_files), )))
        masks_img = resolution_match(masks, data_util, coord1, coord2)
        images_img = resolution_match(images, data_util, coord1, coord2)
    
    else:
        masks_img = sitk.GetImageFromArray(masks)
        images_img = sitk.GetImageFromArray(images)



    sitk.WriteImage(masks_img, os.path.join(path, "3Doutput.nii"))
    sitk.WriteImage(images_img, os.path.join(path, "3Dinput.nii"))
