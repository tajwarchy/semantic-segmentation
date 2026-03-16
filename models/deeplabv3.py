import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ with configurable encoder backbone.

    Architecture:
        - Encoder: ResNet (pretrained on ImageNet)
        - ASPP: Atrous Spatial Pyramid Pooling (captures multi-scale context)
        - Decoder: Combines low-level + high-level features
        - Head: 1x1 conv → num_classes

    Why DeepLabV3+:
        - ASPP lets the model see context at multiple scales simultaneously
        - Outperforms U-Net on large-scale segmentation benchmarks
        - Strong pretrained encoders give a big head start
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        num_classes: int = 21,
        pretrained_encoder: bool = True
    ):
        super().__init__()

        encoder_weights = "imagenet" if pretrained_encoder else None

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_deeplabv3plus(config: dict) -> DeepLabV3Plus:
    model = DeepLabV3Plus(
        encoder_name=config["backbone"],
        num_classes=config["num_classes"],
        pretrained_encoder=config["pretrained_encoder"]
    )
    return model