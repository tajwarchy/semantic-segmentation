import sys
import os
sys.path.append(".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import csv
from PIL import Image

from configs.config_loader import load_config
from models import build_model
from inference.segment import load_model, segment_image
from inference.visualization import (
    mask_to_colormap, blend_overlay,
    add_legend, make_comparison_grid, confidence_heatmap
)
from data.dataset import VOC_CLASSES, VOC_COLORMAP
from training.metrics import SegmentationMetrics
from data.dataloader import get_dataloader
import cv2


# ── 1. Load model ─────────────────────────────────────────────────────────────
config = load_config("configs/config.yaml")
device = torch.device(config["device"])
model  = load_model(config, "weights/best_model.pth", device)

os.makedirs("results/predictions",   exist_ok=True)
os.makedirs("results/metrics",       exist_ok=True)
os.makedirs("results/visualizations",exist_ok=True)


# ── 2. Pick 20 diverse val images ─────────────────────────────────────────────
val_list_path = "data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"
with open(val_list_path) as f:
    all_ids = [l.strip() for l in f.readlines()]

# Sample evenly across the list for diversity
step       = len(all_ids) // 20
sample_ids = [all_ids[i * step] for i in range(20)]
print(f"Generating results for {len(sample_ids)} images...\n")


# ── 3. Run inference + save per-image outputs ──────────────────────────────────
all_classes_seen = set()

for img_id in sample_ids:
    img_path  = f"data/VOCdevkit/VOC2012/JPEGImages/{img_id}.jpg"
    image_rgb = np.array(Image.open(img_path).convert("RGB"))

    pred_mask, logits = segment_image(
        image_rgb, model, device, config["image_size"]
    )

    classes_present = [int(c) for c in np.unique(pred_mask) if c < len(VOC_CLASSES)]
    all_classes_seen.update(classes_present)

    color_mask  = mask_to_colormap(pred_mask)
    overlay     = blend_overlay(image_rgb, color_mask, alpha=0.55)
    heatmap_bgr = confidence_heatmap(logits)
    grid        = make_comparison_grid(image_rgb, color_mask, overlay, heatmap_bgr)
    with_legend = add_legend(overlay, classes_present)

    cv2.imwrite(
        f"results/predictions/{img_id}_grid.jpg",
        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        f"results/predictions/{img_id}_legend.jpg",
        cv2.cvtColor(with_legend, cv2.COLOR_RGB2BGR)
    )

    detected = [VOC_CLASSES[c] for c in classes_present]
    print(f"  ✅ {img_id:<20}  classes: {detected}")

print(f"\n✅ Per-image outputs saved to results/predictions/")


# ── 4. Full validation metrics + per-class IoU CSV ────────────────────────────
print("\nRunning full validation for metrics CSV...")

val_loader = get_dataloader(
    root="data", split="val",
    image_size=config["image_size"],
    batch_size=config["batch_size"],
    num_workers=config["num_workers"]
)

metrics = SegmentationMetrics(config["num_classes"], config["ignore_index"])
model.eval()

from tqdm import tqdm
with torch.no_grad():
    for images, masks in tqdm(val_loader, desc="Computing metrics"):
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        metrics.update(logits, masks)

results      = metrics.compute()
iou_per_class = results["iou_per_class"]

# Save CSV
csv_path = "results/metrics/per_class_iou.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["class_id", "class_name", "iou", "iou_pct"])
    for i, name in enumerate(VOC_CLASSES):
        val = iou_per_class[i]
        writer.writerow([
            i, name,
            f"{val:.4f}" if not np.isnan(val) else "N/A",
            f"{val*100:.2f}%" if not np.isnan(val) else "N/A"
        ])
    writer.writerow([])
    writer.writerow(["", "mIoU",      f"{results['miou']:.4f}",      f"{results['miou']*100:.2f}%"])
    writer.writerow(["", "Mean Dice", f"{results['mean_dice']:.4f}",  f"{results['mean_dice']*100:.2f}%"])
    writer.writerow(["", "Pixel Acc", f"{results['pixel_acc']:.4f}",  f"{results['pixel_acc']*100:.2f}%"])

print(f"✅ Per-class IoU CSV saved to {csv_path}")
metrics.print_class_iou(VOC_CLASSES, results)


# ── 5. Per-class IoU bar chart ─────────────────────────────────────────────────
print("\nGenerating per-class IoU bar chart...")

valid_mask  = ~np.isnan(iou_per_class)
class_names = [VOC_CLASSES[i] for i in range(len(VOC_CLASSES)) if valid_mask[i]]
class_ious  = [iou_per_class[i] * 100 for i in range(len(VOC_CLASSES)) if valid_mask[i]]
colors      = [
    "#{:02x}{:02x}{:02x}".format(*VOC_COLORMAP[i])
    for i in range(len(VOC_CLASSES)) if valid_mask[i]
]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(class_names, class_ious, color=colors, edgecolor="white", height=0.7)

# Value labels on bars
for bar, val in zip(bars, class_ious):
    ax.text(
        bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%", va="center", fontsize=9, color="white"
    )

ax.axvline(x=results["miou"] * 100, color="cyan", linestyle="--",
           linewidth=1.5, label=f"mIoU = {results['miou']*100:.2f}%")
ax.set_xlabel("IoU (%)", color="white")
ax.set_title("Per-Class IoU — DeepLabV3+ ResNet50 on PASCAL VOC 2012",
             color="white", fontsize=13)
ax.set_xlim(0, 105)
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")

fig.patch.set_facecolor("#1e1e1e")
ax.set_facecolor("#2a2a2a")
ax.legend(facecolor="#333", labelcolor="white")

plt.tight_layout()
plt.savefig("results/visualizations/per_class_iou.png", dpi=150,
            facecolor=fig.get_facecolor())
plt.close()
print("✅ Per-class IoU chart saved.")


# ── 6. Class highlighting demo ────────────────────────────────────────────────
print("\nGenerating class highlighting demo...")

# Find an image with multiple interesting classes
highlight_id  = None
highlight_img = None
highlight_mask = None
target_classes = {14, 15, 6, 7}   # motorbike, person, bus, car

for img_id in sample_ids:
    img_path  = f"data/VOCdevkit/VOC2012/JPEGImages/{img_id}.jpg"
    image_rgb = np.array(Image.open(img_path).convert("RGB"))
    pred_mask, _ = segment_image(image_rgb, model, device, config["image_size"])
    present = set(np.unique(pred_mask).tolist())
    if len(present & target_classes) >= 2:
        highlight_id   = img_id
        highlight_img  = image_rgb
        highlight_mask = pred_mask
        break

if highlight_mask is not None:
    classes_in_img = [c for c in np.unique(highlight_mask) if c < len(VOC_CLASSES)]
    n_classes      = min(len(classes_in_img), 6)
    fig, axes      = plt.subplots(1, n_classes + 1, figsize=(4 * (n_classes + 1), 4))
    fig.patch.set_facecolor("#1e1e1e")

    axes[0].imshow(highlight_img)
    axes[0].set_title("Original", color="white", fontsize=11)
    axes[0].axis("off")

    for i, cls_idx in enumerate(classes_in_img[:n_classes]):
        # Highlight only this class, grey out everything else
        isolated = np.zeros((*highlight_mask.shape, 3), dtype=np.uint8)
        grey     = np.full((*highlight_mask.shape, 3), 60, dtype=np.uint8)
        isolated[highlight_mask == cls_idx] = VOC_COLORMAP[cls_idx]
        blended  = np.where(
            (highlight_mask == cls_idx)[..., None],
            isolated, grey
        ).astype(np.uint8)
        final = (0.6 * highlight_img + 0.4 * blended).astype(np.uint8)
        final[highlight_mask == cls_idx] = (
            0.4 * highlight_img[highlight_mask == cls_idx] +
            0.6 * np.array(VOC_COLORMAP[cls_idx])
        ).astype(np.uint8)

        axes[i + 1].imshow(final)
        axes[i + 1].set_title(VOC_CLASSES[cls_idx], color="white", fontsize=11)
        axes[i + 1].axis("off")

    plt.suptitle("Class Isolation Demo", color="white", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("results/visualizations/class_highlighting.png",
                dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"✅ Class highlighting demo saved (source: {highlight_id})")
else:
    print("  No multi-class image found — skipping highlight demo.")

print("\n✅ All results generated successfully.")
print("\nSummary:")
print(f"  results/predictions/     — {len(sample_ids) * 2} image files")
print(f"  results/metrics/         — per_class_iou.csv")
print(f"  results/visualizations/  — per_class_iou.png, class_highlighting.png")