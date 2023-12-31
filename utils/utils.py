import torch
import torchvision
from dataset import CTDataset
from nifti_dataset import nifti_Dataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    # print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    # print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(train_dir, train_maskdir, val_dir, val_mask_dir, batch_size, train_transform, val_transform, num_workers=4, pin_memory=True):
    
    # print(train_maskdir)
    train_ds = nifti_Dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    val_ds = nifti_Dataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
    )

    train_samples = train_ds.__len__()
    # print("Train samples: ", train_samples)
    val_samples = val_ds.__len__()
    # print("Val samples: ", val_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader

def check_accuracy_multiclass(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.unsqueeze(0).to(device)
            preds = model(x)
            _, tags = torch.max(preds, dim=1)
            num_correct += (tags == y).float().sum()
            
            # print(preds.shape)
            # print(y.shape)
            num_pixels += torch.numel(preds)
            # dice_score += (2 * (preds * y).sum()) / (
            #     (preds + y).sum() + 1e-8
            # )
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    # print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().unsqueeze(1).to(device=device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(loader, model, folder="saved_img/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader): 
        torchvision.utils.save_image(x.float(), f"{folder}/image_{idx}.png")

        if idx % 5 != 0:
            continue
        x = x.to(device=device)
        with torch.no_grad():
            preds = model(x)
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1).float(), f"{folder}/mask_{idx}.png")
    model.train()
