import sys
import os
sys.path.append(".")

import torch
import torch.nn as nn
from tqdm import tqdm

from configs.config_loader import load_config
from models import build_model
from data.dataloader import get_dataloader
from data.dataset import VOC_CLASSES
from training.metrics import SegmentationMetrics


def validate(config: dict, model: nn.Module, device: torch.device) -> dict:
    """
    Runs full validation pass.
    Returns dict with miou, mean_dice, pixel_acc, iou_per_class.
    """
    model.eval()
    metrics = SegmentationMetrics(
        num_classes=config["num_classes"],
        ignore_index=config["ignore_index"]
    )

    val_loader = get_dataloader(
        root="data",
        split="val",
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)              # (B, 21, H, W)
            metrics.update(logits, masks)

    results = metrics.compute()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pth checkpoint file")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["device"])

    print(f"Loading checkpoint: {args.checkpoint}")
    model = build_model(config).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    print("✅ Checkpoint loaded.")

    results = validate(config, model, device)

    metrics_obj = SegmentationMetrics(config["num_classes"], config["ignore_index"])
    metrics_obj.print_class_iou(VOC_CLASSES, results)