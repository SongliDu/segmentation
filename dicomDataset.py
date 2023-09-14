import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset



class dicomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        
        image = np.array(Image.open(img_path).convert('L'))
        if self.transform is not None:
            augmentations = self.transform(image=image)
            augmentations = self.transform(image=image)
            image = augmentations['image']
        return image
        return image