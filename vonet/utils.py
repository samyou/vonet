from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_mb(model: nn.Module) -> float:
    return count_parameters(model, trainable_only=False) * 4 / (1024 * 1024)


def resolve_device(device: str) -> torch.device:
    mps_available = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_available:
            return torch.device("mps")
        return torch.device("cpu")

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but no CUDA device is available.")

    if device == "mps" and not mps_available:
        raise ValueError("MPS requested but no Apple Silicon MPS device is available.")

    return torch.device(device)


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
