import torch
from torchmetrics.classification import MulticlassF1Score
from src.preprocessing.labels import num_classes

def compute_metrics(model, dataloader, device, num_classes=19):

    model.eval()

    IGNORE_INDEX = 255

    f1_metric = MulticlassF1Score(
        num_classes=num_classes,
        average="macro"
    ).to(device)

    f1_metric.reset()

    with torch.no_grad():
        for images, masks in dataloader:

            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # Ignore void pixels (NEW IMPORTANT LOGIC)
            valid = (masks != IGNORE_INDEX)

            f1_metric.update(
                preds[valid],
                masks[valid]
            )

    f1 = f1_metric.compute().item()

    return {
        "F1 Score (Macro)": f1
    }
