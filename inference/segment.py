import sys
import os
sys.path.append(".")

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.config_loader import load_config
from models import build_model
from inference.visualization import (
    mask_to_colormap, blend_overlay, add_legend,
    confidence_heatmap, make_comparison_grid
)
from data.dataset import VOC_CLASSES


# ── Preprocessing ─────────────────────────────────────────────────────────────

def get_inference_transform(image_size: int = 512):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


# ── Model loader (cached) ─────────────────────────────────────────────────────

def load_model(config: dict, checkpoint_path: str, device: torch.device):
    model = build_model(config).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    print(f"✅ Model loaded from {checkpoint_path}")
    print(f"   Checkpoint mIoU : {state['miou']*100:.2f}%")
    print(f"   Trained epochs  : {state['epoch'] + 1}")
    return model


# ── Core inference ────────────────────────────────────────────────────────────

def segment_image(
    image_rgb: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    image_size: int = 512
):
    """
    Run segmentation on a single RGB image.
    Args:
        image_rgb : (H, W, 3) uint8 numpy array
    Returns:
        pred_mask  : (H, W) numpy array of class indices
        logits     : (1, num_classes, H, W) raw model output — for confidence map
    """
    orig_h, orig_w = image_rgb.shape[:2]
    transform      = get_inference_transform(image_size)

    # Preprocess
    transformed = transform(image=image_rgb)
    tensor      = transformed["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

    # Forward pass
    with torch.inference_mode():
        logits = model(tensor)                          # (1, 21, 512, 512)

    # Resize logits back to original image dimensions before argmax
    # This gives sharper boundaries than resizing the mask
    logits_resized = F.interpolate(
        logits,
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False
    )

    pred_mask = logits_resized.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)

    return pred_mask, logits_resized


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_segmentation(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    config: dict,
    output_dir: str = "results/predictions",
    show: bool = True
):
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    print(f"\nImage: {image_path}  shape: {image_rgb.shape}")

    # Inference
    pred_mask, logits = segment_image(
        image_rgb, model, device, config["image_size"]
    )

    # Classes present (excluding background optionally)
    classes_present = [int(c) for c in np.unique(pred_mask) if c != 255]
    print(f"Detected classes: {[VOC_CLASSES[c] for c in classes_present]}")

    # Visualizations
    color_mask  = mask_to_colormap(pred_mask)
    overlay     = blend_overlay(image_rgb, color_mask, alpha=0.55)
    heatmap_bgr = confidence_heatmap(logits)
    grid        = make_comparison_grid(image_rgb, color_mask, overlay, heatmap_bgr)
    with_legend = add_legend(overlay, classes_present)

    # Save outputs
    stem = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_grid.jpg"),
        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_overlay_legend.jpg"),
        cv2.cvtColor(with_legend, cv2.COLOR_RGB2BGR)
    )
    cv2.imwrite(
        os.path.join(output_dir, f"{stem}_mask.png"),
        cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    )
    print(f"✅ Saved to {output_dir}/")

    return pred_mask, classes_present


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="weights/best_model.pth")
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument("--output",     type=str, default="results/predictions")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["device"])
    model  = load_model(config, args.checkpoint, device)

    run_segmentation(args.image, model, device, config, args.output)