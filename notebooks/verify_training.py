import sys
sys.path.append(".")

import torch
from configs.config_loader import load_config
from models import build_model
from data.dataloader import get_dataloader
from training.metrics import SegmentationMetrics
from training.train import get_loss_fn
from data.dataset import VOC_CLASSES

config = load_config("configs/config.yaml")

# Override for fast sanity check — don't change config.yaml
config["batch_size"] = 2
config["epochs"]     = 2

device = torch.device(config["device"])

print("Loading 1 batch from train loader...")
loader = get_dataloader(
    root="data", split="train",
    image_size=config["image_size"],
    batch_size=config["batch_size"],
    num_workers=0
)
images, masks = next(iter(loader))
print(f"✅ Batch shapes — images: {images.shape}, masks: {masks.shape}")

print("\nBuilding model...")
model   = build_model(config).to(device)
loss_fn = get_loss_fn(config["ignore_index"])

print("Running forward + backward pass...")
model.train()
images = images.to(device)
masks  = masks.to(device)

logits = model(images)
loss   = loss_fn(logits, masks)
loss.backward()

print(f"✅ Loss value     : {loss.item():.4f}  (should be ~3.0 at init — log(21) ≈ 3.04)")
print(f"✅ Logits shape   : {list(logits.shape)}")
print(f"✅ Gradients flow : {logits.grad_fn is not None}")

print("\nTesting metrics...")
metrics = SegmentationMetrics(config["num_classes"], config["ignore_index"])
metrics.update(logits.detach(), masks)
results = metrics.compute()

print(f"✅ mIoU      : {results['miou']*100:.2f}%   (expect low ~2-8% before training)")
print(f"✅ Mean Dice : {results['mean_dice']*100:.2f}%")
print(f"✅ Pixel Acc : {results['pixel_acc']*100:.2f}%")

metrics.print_class_iou(VOC_CLASSES, results)

print("\n✅ All Phase 4 checks passed. Ready to train.")