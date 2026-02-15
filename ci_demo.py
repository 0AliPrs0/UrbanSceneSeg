"""
Tiny CI segmentation demo.

This is NOT training a full model,
but it simulates the core idea:
image → segmentation mask → metrics.
"""

import numpy as np


def small_segmentation(image_shape=(64, 64)):
    """Generate a fake image + threshold-based mask."""
    image = np.random.randint(0, 255, image_shape, dtype=np.uint8)
    mask = (image > 128).astype(np.uint8)
    return image, mask


def compute_iou(mask):
    """Simple IoU-like metric demo."""
    intersection = np.sum(mask == 1)
    union = mask.size
    return intersection / union


def main():
    print("\n🚀 Running CI Segmentation Demo...\n")

    img, mask = small_segmentation()

    print("Image shape:", img.shape)
    print("Mask values:", np.unique(mask))

    iou_score = compute_iou(mask)
    print(f"Demo IoU score: {iou_score:.3f}")

    assert iou_score > 0, "IoU should never be zero"
    print("\n✅ CI demo finished successfully!\n")


if __name__ == "__main__":
    main()
