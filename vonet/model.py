from __future__ import annotations

import torch
from torch import nn

from vonet.utils import xavier_init


class ConvReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )


class IModule(nn.Module):
    """VoNet I-module.

    The paper describes this as an Inception-style module without the 1x1 and
    5x5 branches, using 3x3 convolutions only.
    """

    def __init__(
        self,
        in_channels: int,
        branch1_channels: int,
        branch2_channels: int,
        pool_proj_channels: int,
    ) -> None:
        super().__init__()
        self.branch1 = ConvReLU(
            in_channels,
            branch1_channels,
            kernel_size=3,
            padding=1,
        )
        self.branch2 = ConvReLU(
            in_channels,
            branch2_channels,
            kernel_size=3,
            padding=1,
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvReLU(in_channels, pool_proj_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x)],
            dim=1,
        )


class FModule(nn.Module):
    """VoNet F-module.

    The paper describes this as a Fire-style module where the e1x1 layer is
    removed. This keeps a squeeze 1x1 convolution followed by an expand 3x3
    convolution.
    """

    def __init__(self, in_channels: int, squeeze_channels: int, expand_channels: int) -> None:
        super().__init__()
        self.squeeze = ConvReLU(in_channels, squeeze_channels, kernel_size=1)
        self.expand = ConvReLU(squeeze_channels, expand_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        x = self.expand(x)
        return x


class VoNet(nn.Module):
    """PyTorch re-implementation of VoNet from ICCIP 2016.

    The paper describes module-level design choices but does not publish full
    layer widths in the text. The default channel sizes in this implementation
    follow the same principles and keep the model close to the paper's reported
    parameter budget (~394k params / ~1.6 MB in FP32).
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.5) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvReLU(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.features = nn.Sequential(
            IModule(64, 64, 64, 32),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FModule(160, 32, 96),
            FModule(96, 32, 96),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            IModule(96, 80, 80, 32),
            FModule(192, 48, 128),
            nn.Dropout(p=dropout),
        )

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.apply(xavier_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
