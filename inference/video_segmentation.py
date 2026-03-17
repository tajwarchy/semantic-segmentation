import sys
import os
sys.path.append(".")

import torch
import numpy as np
import cv2
import time
from PIL import Image

from configs.config_loader import load_config
from models import build_model
from inference.segment import load_model, segment_image
from inference.visualization import mask_to_colormap, blend_overlay, add_legend
from data.dataset import VOC_CLASSES


def run_video_segmentation(
    source,                          # path to video file, or 0 for webcam
    model: torch.nn.Module,
    device: torch.device,
    config: dict,
    output_path: str = "results/predictions/output_video.mp4",
    display: bool = True,
    process_every_n: int = 1,        # process every Nth frame — set to 2 for speed
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    # Video properties
    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo source  : {source}")
    print(f"Resolution    : {orig_w}x{orig_h}")
    print(f"Input FPS     : {fps_in:.1f}")
    print(f"Total frames  : {total_frames}")

    # Output writer — same resolution as input
    out_w, out_h = orig_w * 2, orig_h   # side-by-side: original | overlay
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_in, (out_w, out_h))

    frame_idx    = 0
    fps_values   = []
    last_overlay = None   # reuse last segmentation on skipped frames

    print("\nProcessing — press Q to stop early if displaying live...\n")

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()

        if frame_idx % process_every_n == 0:
            pred_mask, _ = segment_image(
                frame_rgb, model, device, config["image_size"]
            )
            color_mask   = mask_to_colormap(pred_mask)
            last_overlay = blend_overlay(frame_rgb, color_mask, alpha=0.55)

            # Classes present this frame
            classes_present = [int(c) for c in np.unique(pred_mask)]

        # FPS tracking
        elapsed = time.time() - t0
        fps_values.append(1.0 / max(elapsed, 1e-6))
        avg_fps = np.mean(fps_values[-30:])   # rolling 30-frame average

        # Build side-by-side frame
        overlay_bgr  = cv2.cvtColor(last_overlay, cv2.COLOR_RGB2BGR)
        side_by_side = np.hstack([frame_bgr, overlay_bgr])

        # HUD overlay — FPS + frame counter
        cv2.rectangle(side_by_side, (0, 0), (280, 36), (0, 0, 0), -1)
        cv2.putText(
            side_by_side,
            f"FPS: {avg_fps:.1f}  |  Frame: {frame_idx}/{total_frames}",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2
        )

        # Class label strip along the bottom
        _draw_class_strip(side_by_side, classes_present)

        writer.write(side_by_side)

        if display:
            cv2.imshow("Semantic Segmentation", side_by_side)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early by user.")
                break

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  Frame {frame_idx}/{total_frames}  |  Avg FPS: {avg_fps:.1f}")

    cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    print(f"\n✅ Video saved to: {output_path}")
    print(f"   Frames processed : {frame_idx}")
    print(f"   Average FPS      : {np.mean(fps_values):.1f}")


def _draw_class_strip(frame: np.ndarray, classes_present: list):
    """Draw a compact color-coded class strip at the bottom of the frame."""
    H, W  = frame.shape[:2]
    strip_h = 28
    y0      = H - strip_h
    x       = 6

    # Dark background strip
    cv2.rectangle(frame, (0, y0), (W, H), (20, 20, 20), -1)

    from data.dataset import VOC_COLORMAP
    palette = np.array(VOC_COLORMAP, dtype=np.uint8)

    for cls_idx in classes_present:
        if cls_idx >= len(VOC_CLASSES):
            continue
        color_rgb = tuple(palette[cls_idx].tolist())
        color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
        label     = VOC_CLASSES[cls_idx]

        # Swatch
        cv2.rectangle(frame, (x, y0 + 5), (x + 16, y0 + 22), color_bgr, -1)
        # Label
        cv2.putText(frame, label, (x + 20, y0 + 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        x += len(label) * 9 + 32
        if x > W - 100:
            break


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     type=str,  default="0",
                        help="Video file path or 0 for webcam")
    parser.add_argument("--checkpoint", type=str,  default="weights/best_model.pth")
    parser.add_argument("--config",     type=str,  default="configs/config.yaml")
    parser.add_argument("--output",     type=str,  default="results/predictions/output_video.mp4")
    parser.add_argument("--every_n",    type=int,  default=1,
                        help="Process every Nth frame. Use 2 for faster processing.")
    parser.add_argument("--no_display", action="store_true",
                        help="Disable live display window (faster, for headless runs)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config["device"])
    model  = load_model(config, args.checkpoint, device)

    source = int(args.source) if args.source == "0" else args.source

    run_video_segmentation(
        source       = source,
        model        = model,
        device       = device,
        config       = config,
        output_path  = args.output,
        display      = not args.no_display,
        process_every_n = args.every_n,
    )