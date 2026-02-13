import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from src.models.unet_tl import UNetTL
from tqdm.auto import tqdm

def train_model(
    model,
    train_loader,
    val_loader,
    device,
    save_path,
    num_epochs=200,
    patience=6,
    lr=3e-4
):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"Training on device: {device}")

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    amp_enabled = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_loss = float("inf")
    wait = 0

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        # ---------------------
        # TRAIN
        # ---------------------
        model.train()
        running_train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Train", leave=False)

        for images, masks in train_bar:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(images)

                # Safety resize (NEW)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False
                    )

                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ---------------------
        # VALIDATION
        # ---------------------
        model.eval()
        running_val_loss = 0.0

        val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] - Val", leave=False)

        with torch.no_grad():
            for images, masks in val_bar:

                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = model(images)

                    # Safety resize again
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(
                            outputs,
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False
                        )

                    loss = criterion(outputs, masks)

                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # ---------------------
        # EARLY STOPPING (Notebook Behavior)
        # ---------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(model.state_dict(), save_path)
        else:
            wait += 1
            if wait >= patience:
                print("⏹️ Early stopping triggered.")
                # ⚠️ NOTE: no break (matches new notebook behavior)

    return train_losses, val_losses