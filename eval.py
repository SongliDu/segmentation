from dicomDataset import dicomDataset

dicom_dir = 'data/10000_1'


test = CTDataset(
    image_dir=train_dir,
    mask_dir=train_maskdir,
    transform=train_transform,
)
