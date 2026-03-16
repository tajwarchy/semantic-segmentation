import sys
sys.path.append("..")  # so imports resolve from project root

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from data.dataset import VOCSegmentationDataset, VOC_CLASSES, VOC_COLORMAP
from data.dataloader import get_dataloader
import torch

# ── 1. Check raw samples (no transforms) ──────────────────────────────────────
dataset = VOCSegmentationDataset(root="data", split="train")

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Raw Samples — Image | Mask | Overlay", fontsize=14)

for i in range(3):
    img_pil, mask_pil = dataset.get_raw(i)
    img_np   = np.array(img_pil)
    mask_np  = np.array(mask_pil)

    # Build color mask
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(VOC_COLORMAP):
        color_mask[mask_np == cls_idx] = color

    # Overlay
    overlay = (0.55 * img_np + 0.45 * color_mask).astype(np.uint8)

    axes[i][0].imshow(img_np);        axes[i][0].set_title("Image");   axes[i][0].axis("off")
    axes[i][1].imshow(color_mask);    axes[i][1].set_title("Mask");    axes[i][1].axis("off")
    axes[i][2].imshow(overlay);       axes[i][2].set_title("Overlay"); axes[i][2].axis("off")

plt.tight_layout()
plt.savefig("results/visualizations/raw_samples.png", dpi=150)
plt.show()
print("✅ Raw sample visualization saved.")

# ── 2. Check a transformed batch ──────────────────────────────────────────────
loader = get_dataloader(root="data", split="train", batch_size=4)
images, masks = next(iter(loader))

print(f"\n✅ Batch loaded successfully:")
print(f"   images shape : {images.shape}   dtype: {images.dtype}")
print(f"   masks  shape : {masks.shape}    dtype: {masks.dtype}")
print(f"   image  range : [{images.min():.2f}, {images.max():.2f}]")
print(f"   unique mask values (sample): {masks[0].unique().tolist()}")

# ── 3. Check class distribution in one batch ──────────────────────────────────
print("\nClass distribution in first batch:")
flat = masks[masks != 255].flatten()
unique, counts = torch.unique(flat, return_counts=True)
for cls, cnt in zip(unique.tolist(), counts.tolist()):
    pct = 100 * cnt / flat.numel()
    print(f"   {VOC_CLASSES[cls]:>15s}  ({cls:2d})  {pct:.1f}%")