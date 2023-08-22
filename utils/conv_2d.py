import os 
import numpy as np
import nibabel as nib
from skimage import io
from tqdm import tqdm

ct_dir = '/home/dusongli/project/segmentation/data/COVID-19-CT-Seg_20cases/'
lung_mask_dir = '/home/dusongli/project/segmentation/data/Lung_Mask/'
infection_mask_dir = '/home/dusongli/project/segmentation/data/Infection_Mask/'
lung_and_infection_mask_dir = '/home/dusongli/project/segmentation/data/Lung_and_Infection_Mask/'


ct_2d_dir = '/home/dusongli/project/segmentation/data/2D_CT/'
lung_mask_2d_dir = '/home/dusongli/project/segmentation/data/2D_Mask/'
lung_and_infection_mask_2d_dir = '/home/dusongli/project/segmentation/data/2D_Lung_and_Infection_Mask/'

ct_files = os.listdir(ct_dir)

for ct_file in tqdm(ct_files):
    mask_path = os.path.join(lung_and_infection_mask_dir, ct_file)

    mask_data = nib.load(mask_path).get_fdata()

    for i in range(mask_data.shape[2]):
        
        mask_slice = mask_data[:, :, i]
        # mask_slice = (mask_slice) / 3 * 255
        mask_slice = mask_slice.astype(np.uint8)

        mask_slice_name = f"{os.path.splitext(ct_file)[0]}_slice_{i}.png"        
        mask_slice_path = os.path.join(lung_and_infection_mask_2d_dir, mask_slice_name)
        io.imsave(mask_slice_path, mask_slice)
