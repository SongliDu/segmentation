import os
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib

## dataset for 3D nifti files
class nifti_Dataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.files = os.listdir(image_dir)

        self.images = []
        self.masks = []

        ## convert 3D nifti files to 2D 
        for file in self.files:
            image_path = os.path.join(self.image_dir, file)
            mask_path  = os.path.join(self.mask_dir, file)

            image_data = nib.load(image_path).get_fdata()
            mask_data  = nib.load(mask_path).get_fdata()

            for i in range(image_data.shape[2]):
                slice = image_data[:, :, i]
                slice[slice < -1200] = -1200
                slice[slice > 600] = 600
                self.images.append(slice)

                slice = mask_data[:, :, i]
                slice[slice > 0] = 1.0
                self.masks.append(slice)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.transform is not None:
            augmentations = self.transform(image=self.images[index], mask=self.masks[index])
            image = augmentations['image']
            mask = augmentations['mask']
        return image, mask
