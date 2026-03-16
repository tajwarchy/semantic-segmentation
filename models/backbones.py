# Encoders available via segmentation_models_pytorch
# Format: name -> (params_approx, notes)

AVAILABLE_ENCODERS = {
    "resnet34":  {
        "params":  "21M",
        "speed":   "fast",
        "notes":   "Good for U-Net, fast training on M1"
    },
    "resnet50":  {
        "params":  "25M",
        "speed":   "medium",
        "notes":   "Best balance for DeepLabV3+"
    },
    "resnet101": {
        "params":  "44M",
        "speed":   "slow",
        "notes":   "Highest accuracy, use if mIoU stuck below 65%"
    },
    "mobilenet_v2": {
        "params":  "3M",
        "speed":   "very fast",
        "notes":   "Use for real-time inference priority"
    },
}

AVAILABLE_MODELS = ["deeplabv3plus", "unet"]


def validate_config(config: dict):
    model   = config.get("model")
    backbone = config.get("backbone")
    
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model}'. Choose from: {AVAILABLE_MODELS}"
        )
    if backbone not in AVAILABLE_ENCODERS:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Choose from: {list(AVAILABLE_ENCODERS.keys())}"
        )
    return True


def print_encoder_info(encoder_name: str):
    info = AVAILABLE_ENCODERS.get(encoder_name)
    if info:
        print(f"\nEncoder: {encoder_name}")
        for k, v in info.items():
            print(f"  {k:<8}: {v}")
    else:
        print(f"No info found for encoder: {encoder_name}")


def list_encoders():
    print("\nAvailable Encoders:")
    print(f"  {'Name':<16} {'Params':<8} {'Speed':<12} Notes")
    print("  " + "-"*60)
    for name, info in AVAILABLE_ENCODERS.items():
        print(f"  {name:<16} {info['params']:<8} {info['speed']:<12} {info['notes']}")