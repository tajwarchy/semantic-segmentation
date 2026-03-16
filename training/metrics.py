import torch
import numpy as np


class SegmentationMetrics:
    """
    Tracks segmentation metrics across an entire epoch.
    Call update() after each batch, compute() at epoch end.

    Metrics:
        - mIoU       : mean Intersection over Union (primary metric)
        - Dice       : mean Dice coefficient
        - Pixel Acc  : overall pixel accuracy
        - Per-class IoU table
    """

    def __init__(self, num_classes: int = 21, ignore_index: int = 255):
        self.num_classes   = num_classes
        self.ignore_index  = ignore_index
        self.reset()

    def reset(self):
        # Confusion matrix: rows = ground truth, cols = predicted
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds   : (B, num_classes, H, W) raw logits  — or (B, H, W) class indices
            targets : (B, H, W) ground truth class indices, ignore_index for boundaries
        """
        # If preds are logits, convert to class indices
        if preds.dim() == 4:
            preds = preds.argmax(dim=1)   # (B, H, W)

        preds   = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()

        # Remove ignore_index pixels
        valid   = targets != self.ignore_index
        preds   = preds[valid]
        targets = targets[valid]

        # Accumulate into confusion matrix
        # Each (target, pred) pair increments one cell
        np.add.at(
            self.confusion_matrix,
            (targets, preds),
            1
        )

    def compute(self) -> dict:
        cm = self.confusion_matrix.astype(np.float64)

        # Per-class IoU
        # IoU_c = TP_c / (TP_c + FP_c + FN_c)
        #       = cm[c,c] / (col_sum[c] + row_sum[c] - cm[c,c])
        intersection = np.diag(cm)
        union        = cm.sum(axis=1) + cm.sum(axis=0) - intersection

        # Only compute IoU for classes that actually appear in ground truth
        valid_classes = cm.sum(axis=1) > 0
        iou_per_class = np.where(
            valid_classes,
            np.divide(intersection, union, where=union > 0),
            np.nan
        )

        miou = np.nanmean(iou_per_class)

        # Dice per class
        # Dice_c = 2*TP / (2*TP + FP + FN)
        dice_denom    = cm.sum(axis=1) + cm.sum(axis=0)
        dice_per_class = np.where(
            valid_classes,
            np.divide(2 * intersection, dice_denom, where=dice_denom > 0),
            np.nan
        )
        mean_dice = np.nanmean(dice_per_class)

        # Pixel accuracy
        pixel_acc = intersection.sum() / cm.sum() if cm.sum() > 0 else 0.0

        return {
            "miou":          float(miou),
            "mean_dice":     float(mean_dice),
            "pixel_acc":     float(pixel_acc),
            "iou_per_class": iou_per_class,   # numpy array, length num_classes
        }

    def print_class_iou(self, class_names: list, results: dict = None):
        if results is None:
            results = self.compute()

        iou = results["iou_per_class"]
        print("\n" + "="*50)
        print(f"  {'Class':<18} {'IoU':>8}  {'Bar'}")
        print("="*50)
        for i, name in enumerate(class_names):
            val = iou[i]
            if np.isnan(val):
                bar = "  (not present)"
                print(f"  {name:<18} {'N/A':>8}  {bar}")
            else:
                bar = "█" * int(val * 30)
                print(f"  {name:<18} {val*100:>7.1f}%  {bar}")
        print("="*50)
        print(f"  {'mIoU':<18} {results['miou']*100:>7.1f}%")
        print(f"  {'Mean Dice':<18} {results['mean_dice']*100:>7.1f}%")
        print(f"  {'Pixel Acc':<18} {results['pixel_acc']*100:>7.1f}%")
        print("="*50 + "\n")