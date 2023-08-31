from dicomDataset import dicomDataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision
from model.unet import Unet

dicom_dir = 'data/2D_Test'
image_size = 512
output_dir = 'data/test2'


## data loader
transform = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

test_ds = dicomDataset(
    image_dir=dicom_dir,
    transform=transform,
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
)

## model
checkpoint = torch.load('lung_and_infection.pth.tar')
model = Unet(in_channels=1, out_channels=4)
model.load_state_dict(checkpoint['state_dict'])
model.to(device='cuda')
model.eval()

for idx, x in enumerate(test_loader): 
    torchvision.utils.save_image(x.float(), f"{output_dir}/image_{idx}.png")

    x = x.to(device='cuda')
    with torch.no_grad():
        preds = model(x)
        preds = (preds > 0.5).float()
    torchvision.utils.save_image(
        preds, f"{output_dir}/pred_{idx}.png"
    )
model.train()
