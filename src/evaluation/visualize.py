import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from src.preprocessing.mask_utils import decode_segmap
from src.preprocessing.labels import labels, num_classes

def plot_loss_curves(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    plt.title('Training Progress - Loss')
    plt.grid(True, ls='--')
    plt.show()

def visualize_predictions(model, val_dataset, device, save_path=None, num_test_samples=3):
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    print(f'Best model loaded from {save_path}')

    indices_to_test = random.sample(range(len(val_dataset)), num_test_samples)

    fig, axes = plt.subplots(num_test_samples, 3, figsize=(5 * num_test_samples, 5 * num_test_samples))
    if num_test_samples == 1: 
        axes = np.expand_dims(axes, axis=0)

    for i, idx in enumerate(indices_to_test):
        img_tensor, mask_tensor = val_dataset[idx] 

        # Original Image (Denormalize for display)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        # Ground Truth
        gt_colored = decode_segmap(mask_tensor.numpy())

        # Prediction
        with torch.no_grad():
            pred_tensor = model(img_tensor.unsqueeze(0).to(device))
            pred_class = pred_tensor.argmax(dim=1).squeeze().cpu().numpy()
            pred_colored = decode_segmap(pred_class)

        # Plotting
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f'Sample {idx}: Image')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_colored)
        axes[i, 1].set_title(f'Sample {idx}: GT Mask')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title(f'Sample {idx}: Prediction')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()
