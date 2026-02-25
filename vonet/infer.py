from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from vonet.constants import CLASS_NAMES
from vonet.data import build_eval_transform
from vonet.model import VoNet
from vonet.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VoNet inference for one image.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--image", type=Path, required=True, help="Path to an RGB image.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--input-size", type=int, default=227)
    return parser.parse_args()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unexpected checkpoint format in {path}")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model = VoNet(num_classes=len(CLASS_NAMES))
    model.load_state_dict(_load_state_dict(args.checkpoint))
    model.eval().to(device)

    transform = build_eval_transform(args.input_size)
    with Image.open(args.image) as image:
        image = image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_idx = int(torch.argmax(probs).item())

    print(f"device: {device}")
    print(f"predicted_class: {CLASS_NAMES[pred_idx]}")
    print(f"confidence: {float(probs[pred_idx]):.4f}")
    print("probabilities:")
    for class_name, prob in zip(CLASS_NAMES, probs.tolist()):
        print(f"  {class_name:>10}: {prob:.4f}")


if __name__ == "__main__":
    main()
