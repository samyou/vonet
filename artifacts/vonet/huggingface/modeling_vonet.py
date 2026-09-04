from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput

from .configuration_vonet import VoNetConfig


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
    def __init__(
        self,
        in_channels: int,
        branch1_channels: int,
        branch2_channels: int,
        pool_proj_channels: int,
    ) -> None:
        super().__init__()
        self.branch1 = ConvReLU(in_channels, branch1_channels, kernel_size=3, padding=1)
        self.branch2 = ConvReLU(in_channels, branch2_channels, kernel_size=3, padding=1)
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvReLU(in_channels, pool_proj_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x)], dim=1)


class FModule(nn.Module):
    def __init__(self, in_channels: int, squeeze_channels: int, expand_channels: int) -> None:
        super().__init__()
        self.squeeze = ConvReLU(in_channels, squeeze_channels, kernel_size=1)
        self.expand = ConvReLU(squeeze_channels, expand_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.squeeze(x))


class VoNetForImageClassification(PreTrainedModel):
    config_class = VoNetConfig
    main_input_name = "pixel_values"

    def __init__(self, config: VoNetConfig) -> None:
        super().__init__(config)
        self.stem = nn.Sequential(
            ConvReLU(config.num_channels, 64, kernel_size=7, stride=2, padding=3),
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
            nn.Dropout(p=config.dropout),
        )
        self.classifier = nn.Conv2d(128, config.num_labels, kernel_size=1, bias=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> ImageClassifierOutput | tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        x = self.stem(pixel_values)
        x = self.features(x)
        x = self.classifier(x)
        logits = torch.flatten(self.avgpool(x), 1)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        return ImageClassifierOutput(loss=loss, logits=logits)
