import os
import platform
from torch.utils.data import DataLoader


def get_dataloaders(train_dataset, val_dataset, device, batch_size=8):

    # OS-aware worker logic
    if platform.system() == "Windows":
        num_workers = 0
    else:
        num_workers = min(4, os.cpu_count() or 2)

    pin_memory = (device.type == "cuda")
    persistent_workers = (num_workers > 0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader
