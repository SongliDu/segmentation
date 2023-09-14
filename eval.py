import torch
from torchvision.transforms import ToTensor
from model.unet import Unet
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib

input_file = "data/test/10000_1.nii.gz"
output_file = "data/test/10000_1_mask180.nii.gz"


## model
checkpoint = torch.load('checkpoint/lung_epoch_180.pth.tar')
model = Unet(in_channels=1, out_channels=1)
model.load_state_dict(checkpoint['state_dict'])
model.to(device='cuda')
model.eval()

# input_image = sitk.ReadImage(input_file)
preds = []

ct_data = nib.load(input_file).get_fdata()

for i in range(ct_data.shape[2]):
    # image = sitk.GetArrayFromImage(input_image)[i]
    image = ct_data[:, :, i]

    image[image < -1200] = -1200
    image[image > 600] = 600
    min = np.min(image)
    max = np.max(image)
    image = (image - min) / (max - min) * 255
    image = image.astype(np.uint8)
    # image = image / 255.0
    

    tensor_image = ToTensor()(image).unsqueeze(0).to(device='cuda').float()

    with torch.no_grad():
        pred = torch.sigmoid(model(tensor_image))
        pred = (pred > 0.5).float()
        pred = pred.cpu().numpy().squeeze(0).squeeze(0)
        # print(pred.shape)
        preds.append(pred)
    
    # plt.subplot(1, 2, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    # plt.subplot(1, 2, 2)
    # plt.imshow(pred, cmap='gray')
    # plt.title('Predicted Mask')
    # plt.show()

preds = np.array(preds, dtype='float32')
print(preds.shape)
sitk.WriteImage(sitk.GetImageFromArray(preds), output_file)