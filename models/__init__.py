from models.deeplabv3 import build_deeplabv3plus
from models.unet import build_unet
from models.backbones import validate_config


def build_model(config: dict):
    """
    Factory function — returns the correct model based on config.
    Usage:
        model = build_model(config)
    """
    validate_config(config)

    model_name = config["model"]

    if model_name == "deeplabv3plus":
        return build_deeplabv3plus(config)
    elif model_name == "unet":
        return build_unet(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")