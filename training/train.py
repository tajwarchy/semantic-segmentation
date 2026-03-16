import sys
import os
sys.path.append(".")

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time

from configs.config_loader import load_config, print_config
from models import build_model
from data.dataloader import get_dataloader
from data.dataset import VOC_CLASSES
from training.validate import validate
from training.metrics import SegmentationMetrics


# ── Loss ──────────────────────────────────────────────────────────────────────

def get_loss_fn(ignore_index: int = 255) -> nn.Module:
    """
    CrossEntropyLoss with ignore_index=255 for VOC boundary pixels.
    Boundary pixels labeled 255 are excluded from gradient updates.
    """
    return nn.CrossEntropyLoss(ignore_index=ignore_index)


# ── Optimizer + Scheduler ─────────────────────────────────────────────────────

def get_optimizer(model: nn.Module, config: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )


def get_scheduler(optimizer, config: dict, steps_per_epoch: int):
    if config["lr_scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"] * steps_per_epoch
        )
    elif config["lr_scheduler"] == "poly":
        total_steps = config["epochs"] * steps_per_epoch
        lam = lambda step: (1 - step / total_steps) ** 0.9
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lam)
    else:
        raise ValueError(f"Unknown scheduler: {config['lr_scheduler']}")


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, miou, config, filename):
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    path = os.path.join(config["checkpoint_dir"], filename)
    torch.save({
        "epoch":               epoch,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "miou":                miou,
        "config":              config,
    }, path)
    return path


def load_checkpoint(model, optimizer, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state["epoch"], state["miou"]


# ── Training Loop ─────────────────────────────────────────────────────────────

def train(config: dict, resume_from: str = None):
    device = torch.device(config["device"])
    print(f"\nDevice: {device}")

    # Data
    train_loader = get_dataloader(
        root="data", split="train",
        image_size=config["image_size"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    # Model, loss, optimizer, scheduler
    model     = build_model(config).to(device)
    loss_fn   = get_loss_fn(config["ignore_index"])
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config, len(train_loader))

    # TensorBoard
    os.makedirs(config["log_dir"], exist_ok=True)
    writer = SummaryWriter(log_dir=config["log_dir"])

    # Resume
    start_epoch = 0
    best_miou   = 0.0
    if resume_from and os.path.exists(resume_from):
        start_epoch, best_miou = load_checkpoint(model, optimizer, resume_from, device)
        start_epoch += 1
        print(f"✅ Resumed from epoch {start_epoch}, best mIoU: {best_miou*100:.2f}%")

    print(f"\nStarting training: {config['epochs']} epochs")
    print(f"Train batches/epoch : {len(train_loader)}")
    print(f"Model               : {config['model']} + {config['backbone']}")
    print(f"Batch size          : {config['batch_size']}")
    print("-" * 50)

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss   = 0.0
        train_metrics = SegmentationMetrics(config["num_classes"], config["ignore_index"])
        t0 = time.time()

        # ── Train one epoch ──
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)                  # (B, 21, H, W)
            loss   = loss_fn(logits, masks)

            loss.backward()

            # Gradient clipping — stabilises early training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            train_metrics.update(logits.detach(), masks)

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        # ── Epoch stats ──
        avg_loss     = epoch_loss / len(train_loader)
        train_results = train_metrics.compute()
        elapsed      = time.time() - t0

        writer.add_scalar("Loss/train",      avg_loss,                    epoch)
        writer.add_scalar("mIoU/train",      train_results["miou"],       epoch)
        writer.add_scalar("Dice/train",      train_results["mean_dice"],  epoch)
        writer.add_scalar("PixelAcc/train",  train_results["pixel_acc"],  epoch)
        writer.add_scalar("LR",              optimizer.param_groups[0]["lr"], epoch)

        print(f"\nEpoch {epoch+1:>3}/{config['epochs']} "
              f"| Loss: {avg_loss:.4f} "
              f"| Train mIoU: {train_results['miou']*100:.2f}% "
              f"| LR: {optimizer.param_groups[0]['lr']:.6f} "
              f"| Time: {elapsed:.0f}s")

        # ── Validation ──
        if (epoch + 1) % config["log_every_n_epochs"] == 0 or epoch == config["epochs"] - 1:
            print("  Running validation...")
            val_results = validate(config, model, device)

            writer.add_scalar("mIoU/val",     val_results["miou"],      epoch)
            writer.add_scalar("Dice/val",     val_results["mean_dice"], epoch)
            writer.add_scalar("PixelAcc/val", val_results["pixel_acc"], epoch)

            print(f"  Val mIoU     : {val_results['miou']*100:.2f}%")
            print(f"  Val Dice     : {val_results['mean_dice']*100:.2f}%")
            print(f"  Val Pix Acc  : {val_results['pixel_acc']*100:.2f}%")

            # Save best
            if val_results["miou"] > best_miou:
                best_miou = val_results["miou"]
                path = save_checkpoint(
                    model, optimizer, epoch,
                    best_miou, config, "best_model.pth"
                )
                print(f"  ✅ New best mIoU: {best_miou*100:.2f}% — saved to {path}")

        # Always save latest checkpoint (for resuming)
        save_checkpoint(
            model, optimizer, epoch,
            best_miou, config, "latest.pth"
        )

    writer.close()
    print(f"\n{'='*50}")
    print(f"Training complete. Best val mIoU: {best_miou*100:.2f}%")
    print(f"Best model saved at: {config['checkpoint_dir']}/best_model.pth")
    print(f"{'='*50}")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    print_config(config)
    train(config, resume_from=args.resume)