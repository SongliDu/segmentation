import os 
import numpy as np
import nibabel as nib
from skimage import io
from tqdm import tqdm

# ct_dir = '/home/dusongli/project/segmentation/data/COVID-19-CT-Seg_20cases/'
# lung_mask_dir = '/home/dusongli/project/segmentation/data/Lung_Mask/'
# infection_mask_dir = '/home/dusongli/project/segmentation/data/Infection_Mask/'
# lung_and_infection_mask_dir = '/home/dusongli/project/segmentation/data/Lung_and_Infection_Mask/'


# ct_2d_dir = '/home/dusongli/project/segmentation/data/2D_CT/'
# lung_mask_2d_dir = '/home/dusongli/project/segmentation/data/2D_Mask/'
# lung_and_infection_mask_2d_dir = '/home/dusongli/project/segmentation/data/2D_Lung_and_Infection_Mask/'


test_file = 'data/10000_1.nii.gz'
test_out_dir ='data/2D_Test'
test_data = nib.load(test_file).get_fdata()

for i in range(test_data.shape[2]):
    
    image = test_data[:, :, i]
    image[image < -1200] = -1200
    image[image > 600] = 600
    # mask_slice = (mask_slice) / 3 * 255
    min = np.min(image)
    max = np.max(image)
    image = (image - min) / (max - min) * 255

    image = image.astype(np.uint8)

    image_name = f"slice_{i}.png"        
    image_path = os.path.join(test_out_dir, image_name)
    io.imsave(image_path, image)



# for ct_file in tqdm(ct_files):
#     mask_path = os.path.join(lung_and_infection_mask_dir, ct_file)

#     mask_data = nib.load(mask_path).get_fdata()

#     for i in range(mask_data.shape[2]):
        
#         mask_slice = mask_data[:, :, i]
#         mask_slice = mask_slice
#         # mask_slice = (mask_slice) / 3 * 255

#         mask_slice = mask_slice.astype(np.uint8)

#         mask_slice_name = f"{os.path.splitext(ct_file)[0]}_slice_{i}.png"        
#         mask_slice_path = os.path.join(lung_and_infection_mask_2d_dir, mask_slice_name)
#         io.imsave(mask_slice_path, mask_slice)



# import pydicom
# import os
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# dicom_dir = 'data/10000_1'
# output_dir = 'data/test'
# dicom_files = os.listdir(dicom_dir)
# image_files = os.listdir(output_dir)
# dicom_file = os.path.join(dicom_dir, dicom_files[0])
# ds = pydicom.dcmread(dicom_file)
# image_arr = ds.pixel_array
# image = np.array(Image.open(os.path.join(output_dir, image_files[0])).convert('L'))
# for i in range (512):
#     print(image[i])
# plt.imshow(image, cmap='gray')
# plt.show()


# for i in dicom_files:
#     dicom_file = os.path.join(dicom_dir, i)
#     ds = pydicom.dcmread(dicom_file)
#     image_arr = ds.pixel_array
#     image = Image.fromarray(image_arr)

#     image_name = f"{os.path.splitext(i)[0]}.png"
#     image_path = os.path.join(output_dir, image_name)
#     image.save(image_path)