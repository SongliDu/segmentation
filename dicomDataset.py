import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pydicom

class dicomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        ds = pydicom.dcmread(img_path)
        image = ds.pixel_array
        image[image == -2000] = 25
        min = np.min(image)
        max = np.max(image)
        image = (image - min) / (max - min) * 255
        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations['image']
        return image
    
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