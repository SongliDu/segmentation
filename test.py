from model.unet import Unet  
from torchvision.transforms import ToTensor
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import nibabel as nib


# def main():
#     image_path = '/home/dusongli/project/segmentation/data/2D_Val/coronacases_001.nii_slice_0.png'  
#     # mask_path = '/home/dusongli/project/segmentation/data/2D_Lung_and_Infection_Mask/coronacases_001.nii_slice_100.png'  
#     mask_path = '/home/dusongli/project/segmentation/data/2D_Val_Mask/coronacases_001.nii_slice_0.png'
#     image2_path = '/home/dusongli/project/segmentation/data/test/image_2.png'
#     dicom_file = 'data/10000_1/M55264A0'

#     image = np.array(Image.open(image_path))
#     mask = np.array(Image.open(mask_path), dtype=np.float32)
    
#     output_folder = 'saved_img/'

#     checkpoint = torch.load('lung_epoch_25.pth.tar')
#     model = Unet(in_channels=1, out_channels=1)
#     model.load_state_dict(checkpoint['state_dict'])
#     model.to(device='cuda')
#     model.eval()
#     input = ToTensor()(image).unsqueeze(0).to(device='cuda').float()

#     with torch.no_grad():
#         pred = torch.sigmoid(model(input))
#         pred = (pred > 0.5).float()
#         pred = pred.cpu().numpy().squeeze(0).squeeze(0)

#     # torchvision.utils.save_image(pred, f"{output_folder}/prediction.png")
#     # prediction = np.array(Image.open(f"{output_folder}/prediction.png").convert("L"), dtype=np.float32)
#     plt.subplot(1,2,1)
#     plt.imshow(image, cmap='gray')
#     plt.subplot(1,2,2)
#     # plt.imshow(mask, cmap='gray')
#     # plt.subplot(1,3,3)
#     plt.imshow(pred, cmap='gray')

#     plt.show()


ct_file = 'data/test/coronacases_001.nii.gz'

def main():
    ct_data = nib.load(ct_file).get_fdata()
    slice = ct_data[:, :, 0]
    slice[slice < -1200] = -1200
    slice[slice > 600] = 600
    min = np.min(slice)
    max = np.max(slice)
    slice = (slice - min) / (max - min) * 255

    slice = slice.astype(np.uint8)

    image = slice

    checkpoint = torch.load('lung_epoch_25.pth.tar')
    model = Unet(in_channels=1, out_channels=1)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device='cuda')
    model.eval()
    input = ToTensor()(image).unsqueeze(0).to(device='cuda').float()

    with torch.no_grad():
        pred = torch.sigmoid(model(input))
        pred = (pred > 0.5).float()
        pred = pred.cpu().numpy().squeeze(0).squeeze(0)

    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(pred, cmap='gray')
    plt.show()



if __name__ == "__main__":
    main()