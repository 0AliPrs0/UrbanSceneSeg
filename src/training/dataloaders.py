from torch.utils.data import DataLoader
from src.data.cityscapes_dataset import CityscapesDataset
from src.preprocessing.labels import num_classes

# Dataset paths and image/mask lists need to be passed from main
def create_dataloaders(train_images_folder_path, train_mask_folder_path, train_imgs, train_msks,
                       val_images_folder_path, val_mask_folder_path, val_imgs, val_msks,
                       batch_size=8, num_workers=2, pin_memory=True):
    
    # Create datasets
    train_dataset = CityscapesDataset(train_images_folder_path, train_mask_folder_path,
                                      train_imgs, train_msks, augment=True)
    val_dataset = CityscapesDataset(val_images_folder_path, val_mask_folder_path,
                                    val_imgs, val_msks, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    print('Data loaders initialized.')
    return train_loader, val_loader
