import sys
sys.path.append(".")

import torch
from configs.config_loader import load_config, print_config
from models import build_model
from models.backbones import list_encoders


# ── 1. Load config ─────────────────────────────────────────────────────────
config = load_config("configs/config.yaml")
print_config(config)

# ── 2. List available encoders ─────────────────────────────────────────────
list_encoders()

# ── 3. Build model ─────────────────────────────────────────────────────────
print(f"\nBuilding model: {config['model']} + {config['backbone']}...")
model = build_model(config)
print("✅ Model built successfully.")

# ── 4. Move to device ──────────────────────────────────────────────────────
device = torch.device(config["device"])
model = model.to(device)
print(f"✅ Model moved to device: {device}")

# ── 5. Forward pass with dummy input ───────────────────────────────────────
B, C, H, W = 2, 3, 512, 512
dummy_input = torch.randn(B, C, H, W).to(device)

print(f"\nRunning forward pass — input shape: {list(dummy_input.shape)}")

model.eval()
with torch.no_grad():
    output = model(dummy_input)

print(f"✅ Forward pass successful.")
print(f"   Input  shape : {list(dummy_input.shape)}")
print(f"   Output shape : {list(output.shape)}")
print(f"   Expected     : [{B}, {config['num_classes']}, {H}, {W}]")

assert output.shape == (B, config["num_classes"], H, W), "❌ Output shape mismatch!"
print(f"\n✅ Output shape verified — [{B}, 21, 512, 512] ✓")

# ── 6. Parameter count ─────────────────────────────────────────────────────
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters:")
print(f"   Total     : {total:,}")
print(f"   Trainable : {trainable:,}")

# ── 7. Quick test with both models ─────────────────────────────────────────
print("\n--- Testing both architectures ---")

for model_name, backbone in [("unet", "resnet34"), ("deeplabv3plus", "resnet50")]:
    cfg = config.copy()
    cfg["model"]   = model_name
    cfg["backbone"] = backbone

    m = build_model(cfg).to(device)
    m.eval()

    with torch.no_grad():
        # Use smaller batch for memory
        x = torch.randn(1, 3, 512, 512).to(device)
        out = m(x)

    params = sum(p.numel() for p in m.parameters())
    print(f"  ✅ {model_name:<14} + {backbone:<12}  "
          f"output: {list(out.shape)}  params: {params:,}")