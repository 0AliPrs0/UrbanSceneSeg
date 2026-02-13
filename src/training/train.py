import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from src.models.unet_tl import UNetTL

def train_model(train_loader, val_loader, device, num_classes=19,
                num_epochs=300, patience=6, save_path='../models/best_unet_cityscapes.pth'):
    
    model = UNetTL(n_classes=num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)
    scaler = torch.cuda.amp.GradScaler()()

    best_val_loss = float('inf')
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Best model saved: {save_path}")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("⏹️ Early stopping triggered.")
                break

    print("🏁 Training completed successfully.")
    return model, train_losses, val_losses
