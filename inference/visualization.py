import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

from data.dataset import VOC_CLASSES, VOC_COLORMAP


# ── Color palette as numpy array for fast indexing ────────────────────────────
PALETTE = np.array(VOC_COLORMAP, dtype=np.uint8)   # (21, 3)


def mask_to_colormap(mask: np.ndarray) -> np.ndarray:
    """
    Convert class index mask → RGB color image.
    Args:
        mask : (H, W) numpy array of class indices (0–20, 255 ignored)
    Returns:
        (H, W, 3) uint8 RGB image
    """
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx in range(len(VOC_CLASSES)):
        color_mask[mask == cls_idx] = PALETTE[cls_idx]
    return color_mask


def blend_overlay(image: np.ndarray, color_mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Blend original image with color segmentation mask.
    Args:
        image      : (H, W, 3) uint8 RGB
        color_mask : (H, W, 3) uint8 RGB from mask_to_colormap()
        alpha      : weight of original image (0.0 = full mask, 1.0 = full image)
    Returns:
        (H, W, 3) uint8 blended image
    """
    overlay = (alpha * image + (1 - alpha) * color_mask).astype(np.uint8)
    return overlay


def add_legend(image: np.ndarray, classes_present: list) -> np.ndarray:
    """
    Add a color-coded class legend panel to the right of the image.
    Args:
        image           : (H, W, 3) uint8 RGB
        classes_present : list of class indices present in this image
    Returns:
        (H, W + legend_width, 3) uint8 RGB
    """
    H, W = image.shape[:2]

    # Legend panel dimensions
    legend_width  = 220
    row_height    = 28
    padding       = 10
    swatch_size   = 18
    font_size     = 16

    legend_h = max(H, padding + len(classes_present) * row_height + padding)
    legend   = np.ones((legend_h, legend_width, 3), dtype=np.uint8) * 30  # dark bg

    pil_legend = Image.fromarray(legend)
    draw       = ImageDraw.Draw(pil_legend)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()

    draw.text((padding, padding - 4), "Classes", fill=(200, 200, 200), font=font)

    for i, cls_idx in enumerate(classes_present):
        y = padding + 24 + i * row_height
        color = tuple(PALETTE[cls_idx].tolist())
        # Color swatch
        draw.rectangle([padding, y, padding + swatch_size, y + swatch_size], fill=color)
        # Class name
        draw.text((padding + swatch_size + 8, y), VOC_CLASSES[cls_idx],
                  fill=(220, 220, 220), font=font)

    legend = np.array(pil_legend)

    # Pad image height if legend is taller
    if legend_h > H:
        pad  = np.zeros((legend_h - H, W, 3), dtype=np.uint8)
        image = np.vstack([image, pad])

    return np.hstack([image, legend[:H if legend_h <= H else legend_h, :]])


def confidence_heatmap(logits: torch.Tensor) -> np.ndarray:
    """
    Visualize per-pixel confidence as a heatmap.
    High confidence = bright, low confidence = dark blue.
    Args:
        logits : (1, num_classes, H, W) raw model output
    Returns:
        (H, W, 3) uint8 BGR heatmap (OpenCV format)
    """
    probs      = F.softmax(logits, dim=1)           # (1, C, H, W)
    confidence = probs.max(dim=1).values            # (1, H, W)
    conf_np    = confidence.squeeze(0).cpu().numpy() # (H, W)

    conf_uint8 = (conf_np * 255).astype(np.uint8)
    heatmap    = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_INFERNO)  # BGR
    return heatmap


def make_comparison_grid(
    original: np.ndarray,
    color_mask: np.ndarray,
    overlay: np.ndarray,
    heatmap_bgr: np.ndarray = None
) -> np.ndarray:
    """
    Create a side-by-side comparison grid.
    Layout: [Original | Mask | Overlay] or [Original | Mask | Overlay | Heatmap]
    """
    H, W = original.shape[:2]

    def add_label(img, text):
        out  = img.copy()
        pil  = Image.fromarray(out)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
        except Exception:
            font = ImageFont.load_default()
        draw.rectangle([0, 0, W, 28], fill=(0, 0, 0))
        draw.text((8, 5), text, fill=(255, 255, 255), font=font)
        return np.array(pil)

    panels = [
        add_label(original,   "Original"),
        add_label(color_mask, "Segmentation"),
        add_label(overlay,    "Overlay"),
    ]

    if heatmap_bgr is not None:
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        heatmap_rgb = cv2.resize(heatmap_rgb, (W, H))
        panels.append(add_label(heatmap_rgb, "Confidence"))

    return np.hstack(panels)