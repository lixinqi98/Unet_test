import numpy as np
from glob import glob
import os
import monai
import torch
from torch.utils.data import DataLoader
import SimpleITK as sitk
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, Transpose,Activations,AsDiscrete, RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
    Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)

class Data():
    def __init__(self, resolution=(3.5714, 3.5714), patch_size=(64, 64)) -> None:
        self.resolution = resolution
        self.patch_size = patch_size
        self.init_train_transform()
        self.init_val_transforms()
        self.init_test_transforms()

    def init_train_transform(self):   
        train = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # CT HU filter
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            CropForegroundd(keys=['image', 'label'], source_key='image'),               # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                          # augmentation
            ScaleIntensityd(keys=['image']),                                              # intensity
            Spacingd(keys=['image', 'label'], pixdim=self.resolution, mode=('bilinear', 'nearest')),  # resolution

            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=1),
            RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=0),
            # RandFlipd(keys=['image', 'label'], prob=0.15, spatial_axis=2),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 36, np.pi * 2), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 36, np.pi / 2, np.pi / 36), padding_mode="zeros"),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                        rotate_range=(np.pi / 2, np.pi / 36, np.pi / 36), padding_mode="zeros"),
            Rand3DElasticd(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=0.1,
                            sigma_range=(5, 8), magnitude_range=(100, 200), scale_range=(0.15, 0.15, 0.15),
                            padding_mode="zeros"),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5, 1.15), sigma_y=(0.5, 1.15), sigma_z=(0.5, 1.15), prob=0.1,),
            RandAdjustContrastd(keys=['image'], gamma=(0.5, 2.5), prob=0.1),
            RandGaussianNoised(keys=['image'], prob=0.1, mean=np.random.uniform(0, 0.5), std=np.random.uniform(0, 15)),
            RandShiftIntensityd(keys=['image'], offsets=np.random.uniform(0,0.3), prob=0.1),

            SpatialPadd(keys=['image', 'label'], spatial_size=self.patch_size, method= 'end'),  # pad if the image is smaller than patch
            RandSpatialCropd(keys=['image', 'label'], roi_size=self.patch_size, random_size=False),
            ToTensord(keys=['image', 'label'])
            ]
        self.train_transforms = Compose(train)
    
    def init_val_transforms(self):
        val = [
            LoadImaged(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),
            # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
            CropForegroundd(keys=['image', 'label'], source_key='image'),                   # crop CropForeground

            NormalizeIntensityd(keys=['image']),                                      # intensity
            ScaleIntensityd(keys=['image']),
            Spacingd(keys=['image', 'label'], pixdim=self.resolution, mode=('bilinear', 'nearest')),  # resolution

            SpatialPadd(keys=['image', 'label'], spatial_size=self.patch_size, method= 'end'),  # pad if the image is smaller than patch
            ToTensord(keys=['image', 'label'])
        ]
        self.val_transforms = Compose(val)
    

    def init_test_transforms(self):
        test = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                # ThresholdIntensityd(keys=['image'], threshold=-135, above=True, cval=-135),  # Threshold CT
                # ThresholdIntensityd(keys=['image'], threshold=215, above=False, cval=215),
                CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground

                NormalizeIntensityd(keys=['image']),  # intensity
                ScaleIntensityd(keys=['image']),
                Spacingd(keys=['image'], pixdim=self.resolution, mode=('bilinear')),  # resolution

                SpatialPadd(keys=['image'], spatial_size=self.patch_size, method= 'end'),  # pad if the image is smaller than patch
                ToTensord(keys=['image'])])
        self.test_transforms = Compose(test)

    
    def init_reverse(self, crop_shape):
        revert = Compose([
            LoadImaged(keys=['image']),
            AddChanneld(keys=['image']),
            Spacingd(keys=['image'], pixdim=self.resolution, mode=('nearest')),
            Resized(keys=['image'], spatial_size=crop_shape, mode=('nearest')),
        ])
        self.reverse_transforms = revert
    
    def load(self, image_path, label_path):
        images = sorted(glob(os.path.join(image_path, '*.nii')))
        segs = sorted(glob(os.path.join(label_path, '*.nii')))
        return images, segs

    def statistics_crop(self, image, resolution):

        files = [{"image": image}]

        reader = sitk.ImageFileReader()
        reader.SetFileName(image)
        image_itk = reader.Execute()
        original_resolution = image_itk.GetSpacing()

        # original size
        transforms = Compose([
            LoadImaged(keys=['image']),
            AddChanneld(keys=['image']),
            ToTensord(keys=['image'])])
        data = monai.data.Dataset(data=files, transform=transforms)
        loader = DataLoader(data, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
        loader = monai.utils.misc.first(loader)
        im, = (loader['image'][0])
        vol = im.numpy()
        original_shape = vol.shape

        # cropped foreground size
        transforms = Compose([
            LoadImaged(keys=['image']),
            AddChanneld(keys=['image']),
            CropForegroundd(keys=['image'], source_key='image', start_coord_key='foreground_start_coord',
                            end_coord_key='foreground_end_coord', ),  # crop CropForeground
            ToTensord(keys=['image', 'foreground_start_coord', 'foreground_end_coord'])])

        data = monai.data.Dataset(data=files, transform=transforms)
        loader = DataLoader(data, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
        loader = monai.utils.misc.first(loader)
        im, coord1, coord2 = (loader['image'][0], loader['foreground_start_coord'][0], loader['foreground_end_coord'][0])
        vol = im[0].numpy()
        coord1 = coord1.numpy()
        coord2 = coord2.numpy()
        crop_shape = vol.shape

        if resolution is not None:

            transforms = Compose([
                LoadImaged(keys=['image']),
                AddChanneld(keys=['image']),
                CropForegroundd(keys=['image'], source_key='image'),  # crop CropForeground
                Spacingd(keys=['image'], pixdim=resolution, mode=('bilinear')),  # resolution
                ToTensord(keys=['image'])])

            data = monai.data.Dataset(data=files, transform=transforms)
            loader = DataLoader(data, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
            loader = monai.utils.misc.first(loader)
            im, = (loader['image'][0])
            vol = im.numpy()
            resampled_size = vol.shape

        else:

            resampled_size = original_shape

        return original_shape, crop_shape, coord1, coord2, resampled_size, original_resolution
    

    def from_numpy_to_itk(self, image_np, image_itk):

        # read image file
        reader = sitk.ImageFileReader()
        reader.SetFileName(image_itk)
        image_itk = reader.Execute()

        image_np = np.transpose(image_np, (1, 0))
        image = sitk.GetImageFromArray(image_np)
        image.SetDirection(image_itk.GetDirection())
        image.SetSpacing(image_itk.GetSpacing())
        image.SetOrigin(image_itk.GetOrigin())
        return image
