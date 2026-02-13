import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy, MulticlassDiceScore
from src.preprocessing.labels import num_classes

def compute_metrics(model, val_loader, device):
    model.eval()

    iou_metric = MulticlassJaccardIndex(num_classes=num_classes, average='macro').to(device)
    pixel_acc_metric = MulticlassAccuracy(num_classes=num_classes, average='micro').to(device)
    dice_metric = MulticlassDiceScore(num_classes=num_classes, average='macro').to(device)

    val_preds, val_targets = [], []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_preds.append(preds)
            val_targets.append(masks)

    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)

    iou = iou_metric(val_preds, val_targets).item()
    pixel_acc = pixel_acc_metric(val_preds, val_targets).item()
    dice = dice_metric(val_preds, val_targets).item()

    metrics = {'IoU': iou, 'Pixel Accuracy': pixel_acc, 'Dice Score': dice}
    return metrics