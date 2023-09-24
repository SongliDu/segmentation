import os 
import numpy as np
import nibabel as nib
from skimage import io
from tqdm import tqdm
import matplotlib.pyplot as plt
ct_dir = '/home/dusongli/project/segmentation/data/3Dircadb/patient/'
left_lung_dir = '/home/dusongli/project/segmentation/data/3Dircadb/leftlung/'
right_lung_dir = '/home/dusongli/project/segmentation/data/3Dircadb/rightlung/'

ct_2d_dir = '/home/dusongli/project/segmentation/data/2D_CT/'
mask_2d_dir = '/home/dusongli/project/segmentation/data/2D_Mask/'

ct_files = os.listdir(ct_dir)

for ct_file in tqdm(ct_files):
    ct_path = os.path.join(ct_dir, ct_file)
    ct_data = nib.load(ct_path).get_fdata()

    left_lung_path = os.path.join(left_lung_dir, ct_file)
    left_lung_data = nib.load(left_lung_path).get_fdata()

    right_lung_path = os.path.join(right_lung_dir, ct_file)
    right_lung_data = nib.load(right_lung_path).get_fdata()

    for i in range(ct_data.shape[2]):
        
        slice = ct_data[:, :, i]
        slice[slice < -1200] = -1200
        slice[slice > 600] = 600
        min = np.min(slice)
        max = np.max(slice)
        slice = (slice - min) / (max - min) * 255
        slice = slice.astype(np.uint8)
        slice_name = f"{os.path.splitext(ct_file)[0]}_slice_{i}.png"
        slice_path = os.path.join(ct_2d_dir, slice_name)
        io.imsave(slice_path, slice)


        left_slice = left_lung_data[:, :, i]
        right_slice = right_lung_data[:, :, i]
        mask_slice = left_slice + right_slice
        mask_slice[mask_slice > 0] = 255
        mask_slice = mask_slice.astype(np.uint8)

        slice_name = f"{os.path.splitext(ct_file)[0]}_slice_{i}.png"
        slice_path = os.path.join(mask_2d_dir, slice_name)
        io.imsave(slice_path, mask_slice)