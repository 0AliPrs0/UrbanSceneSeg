import os
import random
import torch

# Dataset & preprocessing
from preprocessing.dataset import CityscapesDataset, process_mask_to_train_id

# Models
from models.unet import UNet, UNetTL

# Training & evaluation
from training.train import train_model
from training.dataloaders import get_dataloaders
from evaluation.metrics import compute_metrics

# Utils
from utils.paths import (
    train_images_folder_path,
    train_mask_folder_path,
    test_images_folder_path,
    test_mask_folder_path
)
from utils.device import get_device


# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(42)
random.seed(42)


# -----------------------------
# Device
# -----------------------------
device = get_device()


# -----------------------------
# Dataset
# -----------------------------
BATCH_SIZE = 8
NUM_CLASSES = 19

train_imgs = sorted([
    img for img in os.listdir(train_images_folder_path)
    if img.endswith(".png")
])

train_msks = sorted([
    img for img in os.listdir(train_mask_folder_path)
    if img.endswith(".png")
])

val_imgs = sorted([
    img for img in os.listdir(test_images_folder_path)
    if img.endswith(".png")
])

val_msks = sorted([
    img for img in os.listdir(test_mask_folder_path)
    if img.endswith(".png")
])

train_dataset = CityscapesDataset(
    train_images_folder_path,
    train_mask_folder_path,
    train_imgs,
    train_msks,
    augment=True
)

val_dataset = CityscapesDataset(
    test_images_folder_path,
    test_mask_folder_path,
    val_imgs,
    val_msks,
    augment=False
)


# -----------------------------
# Dataloaders
# -----------------------------
train_loader, val_loader = get_dataloaders(
    train_dataset,
    val_dataset,
    device,
    batch_size=BATCH_SIZE
)


# -----------------------------
# Model
# -----------------------------
model = UNetTL(n_classes=NUM_CLASSES).to(device)

if device.type == "cuda":
    model = torch.nn.DataParallel(model)



# -----------------------------
# Training
# -----------------------------
save_path = os.path.join("models", "best_unet_cityscapes.pth")

train_losses, val_losses = train_model(
    model,
    train_loader,
    val_loader,
    device,
    save_path,
    num_epochs=200,
    patience=6,
    lr=3e-4
)


# -----------------------------
# Load Best Model
# -----------------------------
state = torch.load(save_path, map_location=device)
model.module.load_state_dict(state)



# -----------------------------
# Evaluation
# -----------------------------
metrics = compute_metrics(
    model,
    val_loader,
    device,
    num_classes=NUM_CLASSES
)

print(metrics)
