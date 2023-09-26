import torch
from torchvision.transforms import ToTensor
from model.unet import Unet
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2

input_file = "data/test/coronacases_001.nii.gz"
output_file = "data/test/coronacases_001_infection.nii.gz"


## model
checkpoint = torch.load('checkpoint/infection_epoch_10.pth.tar')
model = Unet(in_channels=1, out_channels=1)
model.load_state_dict(checkpoint['state_dict'])
model.to(device='cuda')
model.eval()

preds = []

ct_data = nib.load(input_file).get_fdata()

transform = A.Compose(
    [
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)


# plt.subplot(1, 2, 1)
# plt.imshow(image.squeeze(0), cmap='gray')
# plt.title('image')

# plt.subplot(1, 2, 2)
# plt.imshow(mask.squeeze(0), cmap='gray')
# plt.title('mask')

# plt.show()


for i in range(ct_data.shape[2]):
    image = ct_data[:, :, i]

    image[image < -1200] = -1200
    image[image > 600] = 600

    augmentations = transform(image=image)
    image = augmentations['image']
    # print (image.shape)
    # tensor_image = ToTensor()(image).unsqueeze(0).to(device='cuda').float()

    with torch.no_grad():
        pred = torch.sigmoid(model(image.unsqueeze(0).to(device='cuda').float()))
        pred = (pred > 0.5).float()
        pred = pred.cpu().numpy().squeeze(0).squeeze(0)
        preds.append(pred)

preds = np.array(preds, dtype='float32')
print(preds.shape)
sitk.WriteImage(sitk.GetImageFromArray(preds), output_file)