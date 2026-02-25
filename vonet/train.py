from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import cast

import torch
from torch import nn
from tqdm import tqdm

from vonet.constants import CLASS_NAMES
from vonet.data import make_dataloaders
from vonet.model import VoNet
from vonet.utils import count_parameters, estimate_model_size_mb, resolve_device, save_checkpoint, seed_everything

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train VoNet for 5-class vehicle orientation classification."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root (class folders or raw CompCars root).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vonet"),
        help="Directory for checkpoints and logs.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-size", type=int, default=227)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA.")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume.")
    return parser.parse_args()


def _to_serializable_args(args: argparse.Namespace) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def _format_class_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{class_name}: {counts[class_name]}" for class_name in CLASS_NAMES)


def _format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    global_step: int = 0,
) -> tuple[float, float, int]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with tqdm(loader, desc="train", leave=False) as progress:
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = targets.size(0)
            predictions = logits.argmax(dim=1)
            batch_correct = (predictions == targets).sum().item()
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += batch_correct

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

            global_step += 1

    return total_loss / total_samples, total_correct / total_samples, global_step


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with tqdm(loader, desc="val", leave=False) as progress:
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == targets).sum().item()

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            progress.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{avg_acc:.4f}")

    return total_loss / total_samples, total_correct / total_samples


def main() -> None:
    args = parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    use_amp = args.amp and device.type == "cuda"

    print(f"device: {device}")
    print(f"mixed precision: {use_amp}")

    train_loader, val_loader, metadata = make_dataloaders(
        data_root=args.data_root,
        input_size=args.input_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
        pin_memory=device.type == "cuda",
    )

    print(f"split mode: {metadata['mode']}")
    print(f"train samples: {metadata['train_size']}")
    print(f"val samples: {metadata['val_size']}")
    train_class_counts = cast(dict[str, int], metadata["train_class_counts"])
    val_class_counts = cast(dict[str, int], metadata["val_class_counts"])
    print(f"train class counts: {_format_class_counts(train_class_counts)}")
    print(f"val class counts: {_format_class_counts(val_class_counts)}")

    model = VoNet(num_classes=len(CLASS_NAMES), dropout=args.dropout).to(device)
    print(f"parameters: {count_parameters(model):,}")
    print(f"estimated size (fp32): {estimate_model_size_mb(model):.2f} MB")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_val_acc = 0.0
    global_step = 0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_acc = float(checkpoint.get("best_val_acc", 0.0))
        global_step = int(checkpoint.get("global_step", (start_epoch - 1) * len(train_loader)))
        print(f"resumed from checkpoint: {args.resume}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_start_time = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        train_loss, train_acc, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            global_step=global_step,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )
        epoch_duration = time.perf_counter() - epoch_start_time

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"epoch_time={_format_duration(epoch_duration)}"
        )

        checkpoint_payload = {
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": max(best_val_acc, val_acc),
            "class_names": CLASS_NAMES,
            "args": _to_serializable_args(args),
        }

        save_checkpoint(args.output_dir / "last.pt", checkpoint_payload)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.output_dir / "best.pt", checkpoint_payload)
            print(f"new best checkpoint: {args.output_dir / 'best.pt'}")

    total_training_duration = time.perf_counter() - training_start_time
    print(
        f"training finished. best_val_acc={best_val_acc:.4f} "
        f"duration={_format_duration(total_training_duration)}"
    )


if __name__ == "__main__":
    main()
