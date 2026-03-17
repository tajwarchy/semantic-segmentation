import sys
import os
sys.path.append(".")

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from configs.config_loader import load_config
from models import build_model
from inference.segment import load_model
from data.dataloader import get_dataloader
from data.dataset import VOC_CLASSES, VOC_COLORMAP
from training.metrics import SegmentationMetrics


def main():
    config = load_config("configs/config.yaml")
    config["num_workers"] = 0    # M1 fix
    device = torch.device(config["device"])
    model  = load_model(config, "weights/best_model.pth", device)

    print("Running validation pass to compute per-class IoU...")
    val_loader = get_dataloader(
        root="data", split="val",
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=0
    )

    metrics = SegmentationMetrics(config["num_classes"], config["ignore_index"])
    model.eval()

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            masks  = masks.to(device)
            logits = model(images)
            metrics.update(logits, masks)

    results       = metrics.compute()
    iou_per_class = results["iou_per_class"]
    metrics.print_class_iou(VOC_CLASSES, results)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    print("\nGenerating per-class IoU bar chart...")

    valid_mask  = ~np.isnan(iou_per_class)
    class_names = [VOC_CLASSES[i]  for i in range(len(VOC_CLASSES)) if valid_mask[i]]
    class_ious  = [iou_per_class[i] * 100 for i in range(len(VOC_CLASSES)) if valid_mask[i]]
    colors      = [
        "#{:02x}{:02x}{:02x}".format(*VOC_COLORMAP[i])
        for i in range(len(VOC_CLASSES)) if valid_mask[i]
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")

    bars = ax.barh(class_names, class_ious, color=colors, edgecolor="white", height=0.7)

    for bar, val in zip(bars, class_ious):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center", fontsize=9, color="white"
        )

    miou = results["miou"] * 100
    ax.axvline(x=miou, color="cyan", linestyle="--", linewidth=1.5,
               label=f"mIoU = {miou:.2f}%")
    ax.axvline(x=65, color="#ef5350", linestyle=":", linewidth=1.2,
               label="Target: 65%")

    ax.set_xlabel("IoU (%)", color="white")
    ax.set_xlim(0, 108)
    ax.set_title(
        "Per-Class IoU — DeepLabV3+ ResNet50 on PASCAL VOC 2012",
        color="white", fontsize=13, pad=12
    )
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.legend(facecolor="#333", labelcolor="white", fontsize=10)
    ax.grid(axis="x", color="#444", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    os.makedirs("results/visualizations", exist_ok=True)
    out_path = "results/visualizations/per_class_iou.png"
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()

    print(f"✅ Saved to {out_path}")
    if os.path.exists(out_path):
        size_kb = os.path.getsize(out_path) / 1024
        print(f"✅ File confirmed — {size_kb:.1f} KB")
    else:
        print("❌ File still not found — check write permissions on results/")


if __name__ == "__main__":
    main()