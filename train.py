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
)

import argparse

parser = argparse.ArgumentParser(description='PyTorch 2D Lung Segmentation Training')
parser.add_argument('--epoch', type=int, help='epoch number')
args = parser.parse_args()
epoch_num = args.epoch


ct_dir = "data/2D_CT/"
lung_mask_dir = "data/2D_Mask"

val_dir = "data/2D_Val/"
val_lung_mask_dir = "data/2D_Val_Mask"

device = 'cuda'
batch_size = 8
epochs = 10
lr = 1e-4
image_size = 512
load_model = True

def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        data = data.float()
        targets = targets.float().unsqueeze(1).to(device=device)
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
    model = Unet(in_channels=1, out_channels=1).to(device)
    
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scaler = torch.cuda.amp.GradScaler()

    train_loader, val_loader = get_loaders(
        ct_dir, lung_mask_dir, val_dir, val_lung_mask_dir, batch_size, train_transform, val_transform, num_workers=4
    )

    if (load_model):
        load_checkpoint(torch.load("/home/dusongli/project/segmentation/lung_epoch_390.pth.tar"), model)

    for epoch in range(epochs):
        train(train_loader, model, optimizer, loss_fn, scaler)
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=f"lung_epoch_{epoch + 1 + 390}.pth.tar")
            check_accuracy(val_loader, model, device=device)


if __name__ == '__main__':
    main()