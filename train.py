from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch
from model.unet import Unet
from utils.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

ct_dir = "/home/dusongli/project/segmentation/data/2D_CT/"
# lung_mask_dir = "/home/dusongli/project/segmentation/data/2D_Mask"

val_dir = "/home/dusongli/project/segmentation/data/2D_Val/"
val_lung_and_infection_mask_dir = "/home/dusongli/project/segmentation/data/2D_Val_Lung_and_Infection_Mask"

device = 'cuda'
batch_size = 8
epochs = 2
lr = 1e-4
image_size = 512
load_model = False

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)

        targets = targets.long().to(device=device)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        # if (batch_idx == 5):
        #     break


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=[0.0], std=[1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    model = Unet(in_channels=1, out_channels=4).to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(
        ct_dir, lung_and_infection_mask_dir, val_dir, val_lung_and_infection_mask_dir, batch_size, train_transform, val_transform, num_workers=4
    )

    if (load_model):
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    

    for epoch in range(epochs):
        train(train_loader, model, optimizer, loss_fn, scaler)
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename="lung_and_infection.pth.tar")
        check_accuracy(val_loader, model, device=device)
        save_predictions_as_imgs(val_loader, model, folder="saved_img/", device=device)


if __name__ == '__main__':
    main()