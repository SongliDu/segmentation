from model.unet import Unet  
from torchvision.transforms import ToTensor
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def main():
    image_path = '/home/dusongli/project/segmentation/data/2D_CT/coronacases_009.nii_slice_100.png'  
    # mask_path = '/home/dusongli/project/segmentation/data/2D_Lung_and_Infection_Mask/coronacases_001.nii_slice_100.png'  
    mask_path = '/home/dusongli/project/segmentation/data/2D_Mask/coronacases_009.nii_slice_100.png'
    
    image = np.array(Image.open(image_path))
    mask = np.array(Image.open(mask_path), dtype=np.float32)
    # plt.subplot(1,2,1)
    # plt.imshow(image, cmap='gray')
    # plt.subplot(1,2,2)
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    output_folder = 'saved_img/'

    checkpoint = torch.load('lung.pth.tar')
    model = Unet(in_channels=1, out_channels=1)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device='cuda')
    model.eval()
    input = ToTensor()(image).unsqueeze(0).to(device='cuda').float()

    with torch.no_grad():
        pred = torch.sigmoid(model(input))
        pred = (pred > 0.5).float()

    torchvision.utils.save_image(pred, f"{output_folder}/prediction.png")
    prediction = np.array(Image.open(f"{output_folder}/prediction.png").convert("L"), dtype=np.float32)
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(mask, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(prediction, cmap='gray')

    plt.show()
if __name__ == "__main__":
    main()