import os 
import numpy as np
import nibabel as nib
from skimage import io
from tqdm import tqdm

ct_dir = '/home/dusongli/project/segmentation/data/COVID-19-CT-Seg_20cases/'
# lung_mask_dir = '/home/dusongli/project/segmentation/data/Lung_Mask/'
# infection_mask_dir = '/home/dusongli/project/segmentation/data/Infection_Mask/'
# lung_and_infection_mask_dir = '/home/dusongli/project/segmentation/data/Lung_and_Infection_Mask/'


ct_2d_dir = '/home/dusongli/project/segmentation/data/2D_CT/'
# lung_mask_2d_dir = '/home/dusongli/project/segmentation/data/2D_Mask/'
# lung_and_infection_mask_2d_dir = '/home/dusongli/project/segmentation/data/2D_Lung_and_Infection_Mask/'


# test_file = 'data/10000_1.nii.gz'
# test_out_dir ='data/2D_Test'
# test_data = nib.load(test_file).get_fdata()

# for i in range(test_data.shape[2]):
    
#     image = test_data[:, :, i]
#     image[image < -1200] = -1200
#     image[image > 600] = 600
#     # mask_slice = (mask_slice) / 3 * 255
#     min = np.min(image)
#     max = np.max(image)
#     image = (image - min) / (max - min) * 255

#     image = image.astype(np.uint8)

#     image_name = f"slice_{i}.png"        
#     image_path = os.path.join(test_out_dir, image_name)
#     io.imsave(image_path, image)

ct_files = os.listdir(ct_dir)

for ct_file in tqdm(ct_files):
    ct_path = os.path.join(ct_dir, ct_file)

    ct_data = nib.load(ct_path).get_fdata()

    for i in range(ct_data.shape[2]):
        
        slice = ct_data[:, :, i]
        slice[slice < -1200] = -1200
        slice[slice > 600] = 600
        # mask_slice = (mask_slice) / 3 * 255
        min = np.min(slice)
        max = np.max(slice)
        slice = (slice - min) / (max - min) * 255

        slice = slice.astype(np.uint8)

        slice_name = f"{os.path.splitext(ct_file)[0]}_slice_{i}.png"
        slice_path = os.path.join(ct_2d_dir, slice_name)
        io.imsave(slice_path, slice)