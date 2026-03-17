import sys
sys.path.append(".")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

log_dir = "results/metrics"

# Find the event file
event_files = []
for root, dirs, files in os.walk(log_dir):
    for f in files:
        if f.startswith("events.out"):
            event_files.append(os.path.join(root, f))

if not event_files:
    print("No TensorBoard event files found in results/metrics/")
    exit()

ea = EventAccumulator(event_files[0])
ea.Reload()

def get_scalar(tag):
    try:
        events = ea.Scalars(tag)
        steps  = [e.step for e in events]
        values = [e.value for e in events]
        return steps, values
    except KeyError:
        return [], []

# Pull all tracked scalars
train_loss_steps,  train_loss  = get_scalar("Loss/train")
train_miou_steps,  train_miou  = get_scalar("mIoU/train")
val_miou_steps,    val_miou    = get_scalar("mIoU/val")
val_dice_steps,    val_dice    = get_scalar("Dice/val")
val_acc_steps,     val_acc     = get_scalar("PixelAcc/val")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#1e1e1e")

def style_ax(ax, title, ylabel):
    ax.set_facecolor("#2a2a2a")
    ax.set_title(title, color="white", fontsize=12)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(facecolor="#333", labelcolor="white", fontsize=9)
    ax.grid(color="#444", linestyle="--", linewidth=0.5)

# Plot 1 — Loss
axes[0].plot(train_loss_steps, train_loss, color="#4fc3f7", linewidth=2, label="Train Loss")
style_ax(axes[0], "Training Loss", "CrossEntropy Loss")

# Plot 2 — mIoU
axes[1].plot(train_miou_steps, [v * 100 for v in train_miou],
             color="#81c784", linewidth=2, label="Train mIoU")
axes[1].plot(val_miou_steps, [v * 100 for v in val_miou],
             color="#ffb74d", linewidth=2, linestyle="--", label="Val mIoU")
if val_miou:
    best = max(val_miou) * 100
    axes[1].axhline(y=best, color="cyan", linestyle=":", linewidth=1.2,
                    label=f"Best: {best:.2f}%")
    axes[1].axhline(y=65, color="#ef5350", linestyle=":", linewidth=1,
                    label="Target: 65%")
style_ax(axes[1], "mIoU over Epochs", "mIoU (%)")

# Plot 3 — Val metrics
axes[2].plot(val_dice_steps,  [v * 100 for v in val_dice],
             color="#ce93d8", linewidth=2, label="Val Dice")
axes[2].plot(val_acc_steps,   [v * 100 for v in val_acc],
             color="#80deea", linewidth=2, label="Val Pixel Acc")
style_ax(axes[2], "Validation Metrics", "Score (%)")

plt.suptitle("DeepLabV3+ ResNet50 — PASCAL VOC 2012 Training",
             color="white", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("results/visualizations/training_curves.png",
            dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
plt.close()
print("✅ Training curves saved to results/visualizations/training_curves.png")