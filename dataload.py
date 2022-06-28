import numpy as np
from glob import glob
import os
from monai.transforms import (EnsureType, Compose, LoadImaged, AddChanneld, Transpose,Activations,AsDiscrete, RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                              ScaleIntensityd, ToTensord, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
    Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)

class Data():
    def __init__(self, resolution=(3.5714, 3.5714), patch_size=(32, 32)) -> None:
        self.resolution = resolution
        self.patch_size = patch_size
        self.init_train_transform()
        self.init_val_transforms()

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
    
    def load(self, image_path, label_path):
        images = sorted(glob(os.path.join(image_path, '*.nii')))
        segs = sorted(glob(os.path.join(label_path, '*.nii')))
        return images, segs
