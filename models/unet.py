import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNet(nn.Module):
    """
    U-Net with configurable encoder backbone.

    Architecture:
        - Encoder: ResNet (pretrained on ImageNet) — the downsampling path
        - Skip connections: pass encoder feature maps to decoder
        - Decoder: Progressive upsampling with skip connection fusion
        - Head: 1x1 conv → num_classes

    Why U-Net:
        - Skip connections preserve fine spatial detail
        - Trains faster than DeepLabV3+ — good for quick experiments
        - Excellent for structured/boundary-heavy segmentation
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        num_classes: int = 21,
        pretrained_encoder: bool = True
    ):
        super().__init__()

        encoder_weights = "imagenet" if pretrained_encoder else None

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            decoder_channels=(256, 128, 64, 32, 16),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_unet(config: dict) -> UNet:
    model = UNet(
        encoder_name=config["backbone"],
        num_classes=config["num_classes"],
        pretrained_encoder=config["pretrained_encoder"]
    )
    return model