# main.py
import os
import random
import torch
from torch.utils.data import DataLoader

# Preprocessing & Dataset
from preprocessing.mask_utils import process_mask_to_train_id
from data.cityscapes_dataset import CityscapesDataset

# Models
from models.unet import UNet
from models.unet_tl import UNetTL

# Training
from training.train import train_model

# Evaluation
from evaluation.metrics import compute_metrics
from evaluation.visualize import plot_loss_curves, visualize_predictions

# -------------------------------
# Configs
# -------------------------------
train_images_folder_path = "../data/train/images"
train_mask_folder_path = "../data/train/masks"
test_images_folder_path = "../data/val/images"
test_mask_folder_path = "../data/val/masks"

BATCH_SIZE = 8
NUM_CLASSES = 19
NUM_EPOCHS = 300
PATIENCE = 6
SAVE_PATH = '../models/best_unet_cityscapes.pth'

# -------------------------------
# Reproducibility
# -------------------------------
torch.manual_seed(42)
random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------
# Prepare Dataset & Dataloaders
# -------------------------------
train_imgs = sorted([img for img in os.listdir(train_images_folder_path) if img.endswith('.png')])
train_msks = sorted([img for img in os.listdir(train_mask_folder_path) if img.endswith('.png')])
val_imgs = sorted([img for img in os.listdir(test_images_folder_path) if img.endswith('.png')])
val_msks = sorted([img for img in os.listdir(test_mask_folder_path) if img.endswith('.png')])

train_dataset = CityscapesDataset(train_images_folder_path, train_mask_folder_path, train_imgs, train_msks, augment=True)
val_dataset = CityscapesDataset(test_images_folder_path, test_mask_folder_path, val_imgs, val_msks, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -------------------------------
# Initialize Model
# -------------------------------
model = UNetTL(n_classes=NUM_CLASSES).to(device)

# -------------------------------
# Train Model
# -------------------------------
model, train_losses, val_losses = train_model(
    train_loader, val_loader, device,
    num_classes=NUM_CLASSES,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE,
    save_path=SAVE_PATH
)

# -------------------------------
# Evaluate Model
# -------------------------------
metrics = compute_metrics(model, val_loader, device)
print("Validation Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# -------------------------------
# Plot Loss Curves
# -------------------------------
plot_loss_curves(train_losses, val_losses)

# -------------------------------
# Visualize Predictions
# -------------------------------
visualize_predictions(model, val_dataset, device, save_path=SAVE_PATH, num_test_samples=3)
