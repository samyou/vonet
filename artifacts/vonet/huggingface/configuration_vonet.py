from __future__ import annotations

from transformers import PreTrainedConfig


class VoNetConfig(PreTrainedConfig):
    model_type = "vonet"

    def __init__(
        self,
        dropout: float = 0.5,
        input_size: int = 227,
        num_channels: int = 3,
        **kwargs,
    ) -> None:
        self.dropout = dropout
        self.input_size = input_size
        self.num_channels = num_channels
        super().__init__(**kwargs)
