import numpy as np
from PIL import Image
import torch
from src.preprocessing.labels import labels, valid_colors, valid_ids, num_classes

def process_mask_to_train_id(mask_path, height=96, width=256):
    img = Image.open(mask_path).convert('RGB')
    img = img.resize((width, height), resample=Image.NEAREST)
    flat_mask = np.array(img).astype(np.int32).reshape(-1, 3)
    dists = np.sum((flat_mask[:, None, :] - valid_colors[None, :, :]) ** 2, axis=2)
    min_indices = np.argmin(dists, axis=1)
    min_dists = np.min(dists, axis=1)
    final_ids = valid_ids[min_indices]
    final_ids[min_dists > 50] = 255
    return final_ids.reshape(height, width)


def decode_segmap(pred):
    # pred: [H, W] with class indices 0..18, 255
    cmap = np.array([l.color for l in labels])
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for i in range(num_classes):
        out[pred == i] = cmap[i]
    out[pred == 255] = [0, 0, 0]  # void class to black
    return out
