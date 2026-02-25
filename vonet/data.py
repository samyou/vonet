from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vonet.constants import (
    CLASS_NAME_ALIASES,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    VIEWPOINT_TO_CLASS_NAME,
)


def build_train_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class ImageListDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[tuple[Path, int]],
        transform: Callable,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, label


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def _resolve_class_directory(parent: Path, class_name: str) -> Path | None:
    aliases = CLASS_NAME_ALIASES[class_name]
    normalized_aliases = {_normalize_name(alias) for alias in aliases}

    if not parent.exists() or not parent.is_dir():
        return None

    for entry in parent.iterdir():
        if entry.is_dir() and _normalize_name(entry.name) in normalized_aliases:
            return entry
    return None


def _list_images(directory: Path) -> list[Path]:
    images: list[Path] = []
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    images.sort()
    return images


def _collect_samples(root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        class_dir = _resolve_class_directory(root, class_name)
        if class_dir is None:
            raise FileNotFoundError(
                f"Could not find directory for class '{class_name}' under {root}."
            )

        class_images = _list_images(class_dir)
        if not class_images:
            raise ValueError(f"No images found in class directory: {class_dir}")

        samples.extend((image_path, class_index) for image_path in class_images)

    samples.sort(key=lambda item: str(item[0]))
    return samples


def _read_compcars_viewpoint(label_path: Path) -> int:
    with label_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()

    try:
        return int(first_line)
    except ValueError as exc:
        raise ValueError(f"Invalid viewpoint in label file: {label_path}") from exc


def _resolve_compcars_image_from_label(label_root: Path, image_root: Path, label_path: Path) -> Path | None:
    relative_label = label_path.relative_to(label_root)
    base_relative_image = relative_label.with_suffix("")

    for extension in IMAGE_EXTENSIONS:
        image_path = image_root / base_relative_image.with_suffix(extension)
        if image_path.exists() and image_path.is_file():
            return image_path
    return None


def _collect_compcars_samples_from_labels(data_root: Path) -> list[tuple[Path, int]]:
    image_root = data_root / "image"
    label_root = data_root / "label"
    samples: list[tuple[Path, int]] = []

    for label_path in sorted(label_root.rglob("*.txt")):
        viewpoint = _read_compcars_viewpoint(label_path)
        class_name = VIEWPOINT_TO_CLASS_NAME.get(viewpoint)
        if class_name is None:
            continue

        image_path = _resolve_compcars_image_from_label(
            label_root=label_root,
            image_root=image_root,
            label_path=label_path,
        )
        if image_path is None:
            continue

        class_index = CLASS_NAMES.index(class_name)
        samples.append((image_path, class_index))

    if not samples:
        raise ValueError(
            "No valid CompCars orientation samples found under "
            f"{data_root}. Ensure image/ and label/ exist with viewpoint labels 1-5."
        )

    samples.sort(key=lambda item: str(item[0]))
    return samples


def _extract_image_token(line: str) -> str | None:
    for raw_token in line.strip().split():
        token = raw_token.replace("\\", "/")
        lower_token = token.lower()
        if lower_token.endswith(IMAGE_EXTENSIONS):
            return token
    return None


def _normalize_compcars_relative_path(token: str) -> Path:
    token_path = Path(token.strip())
    parts = [part for part in token_path.parts if part not in {"", ".", token_path.anchor}]

    lower_parts = [part.lower() for part in parts]
    if "image" in lower_parts:
        image_index = lower_parts.index("image")
        parts = parts[image_index + 1 :]
    return Path(*parts)


def _resolve_compcars_image_path(image_root: Path, token: str) -> Path | None:
    relative_path = _normalize_compcars_relative_path(token)
    if relative_path.suffix:
        candidate = image_root / relative_path
        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    for extension in IMAGE_EXTENSIONS:
        candidate = image_root / relative_path.with_suffix(extension)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _collect_compcars_samples_from_split_file(data_root: Path, split_file: Path) -> list[tuple[Path, int]]:
    image_root = data_root / "image"
    label_root = data_root / "label"
    samples: list[tuple[Path, int]] = []

    for line in split_file.read_text(encoding="utf-8").splitlines():
        token = _extract_image_token(line)
        if token is None:
            continue

        image_path = _resolve_compcars_image_path(image_root=image_root, token=token)
        if image_path is None:
            continue

        relative_image = image_path.relative_to(image_root)
        label_path = label_root / relative_image.with_suffix(".txt")
        if not label_path.exists() or not label_path.is_file():
            continue

        viewpoint = _read_compcars_viewpoint(label_path)
        class_name = VIEWPOINT_TO_CLASS_NAME.get(viewpoint)
        if class_name is None:
            continue

        class_index = CLASS_NAMES.index(class_name)
        samples.append((image_path, class_index))

    if not samples:
        raise ValueError(f"No valid samples parsed from split file: {split_file}")

    samples.sort(key=lambda item: str(item[0]))
    return samples


def _pick_split_file(files: list[Path], preferred_names: tuple[str, ...]) -> Path | None:
    if not files:
        return None

    by_name = {path.name.lower(): path for path in files}
    for preferred_name in preferred_names:
        if preferred_name in by_name:
            return by_name[preferred_name]
    return sorted(files, key=lambda path: path.name.lower())[0]


def _load_compcars_split_samples(
    data_root: Path,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]] | None:
    split_root = data_root / "train_test_split" / "classification"
    if not split_root.exists() or not split_root.is_dir():
        return None

    txt_files = [path for path in split_root.iterdir() if path.is_file() and path.suffix.lower() == ".txt"]
    if not txt_files:
        return None

    train_candidates = [
        path
        for path in txt_files
        if "train" in path.stem.lower() and "verification" not in path.stem.lower()
    ]
    val_candidates = [
        path
        for path in txt_files
        if any(token in path.stem.lower() for token in ("val", "test"))
        and "verification" not in path.stem.lower()
    ]

    train_file = _pick_split_file(train_candidates, ("train.txt", "train_web.txt"))
    val_file = _pick_split_file(
        val_candidates,
        ("val.txt", "validation.txt", "test.txt", "test_web.txt"),
    )
    if train_file is None or val_file is None:
        return None

    train_samples = _collect_compcars_samples_from_split_file(data_root, train_file)
    val_samples = _collect_compcars_samples_from_split_file(data_root, val_file)
    return train_samples, val_samples


def _stratified_split(
    samples: Sequence[tuple[Path, int]],
    train_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    grouped: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []

    for class_index, class_samples in grouped.items():
        shuffled = class_samples.copy()
        rng.shuffle(shuffled)

        split_at = int(len(shuffled) * train_ratio)
        if len(shuffled) > 1:
            split_at = min(max(1, split_at), len(shuffled) - 1)

        train_samples.extend(shuffled[:split_at])
        val_samples.extend(shuffled[split_at:])

    train_samples.sort(key=lambda item: str(item[0]))
    val_samples.sort(key=lambda item: str(item[0]))
    return train_samples, val_samples


def _class_counts(samples: Sequence[tuple[Path, int]]) -> dict[str, int]:
    counts = {class_name: 0 for class_name in CLASS_NAMES}
    for _, class_index in samples:
        counts[CLASS_NAMES[class_index]] += 1
    return counts


def build_datasets(
    data_root: Path,
    input_size: int,
    train_ratio: float,
    seed: int,
) -> tuple[ImageListDataset, ImageListDataset, dict[str, object]]:
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    compcars_image_dir = data_root / "image"
    compcars_label_dir = data_root / "label"

    if train_dir.exists() and val_dir.exists():
        mode = "pre-split"
        train_samples = _collect_samples(train_dir)
        val_samples = _collect_samples(val_dir)
    elif compcars_image_dir.exists() and compcars_label_dir.exists():
        split_samples = _load_compcars_split_samples(data_root)
        if split_samples is not None:
            mode = "compcars-split"
            train_samples, val_samples = split_samples
        else:
            mode = "compcars-auto-split"
            full_samples = _collect_compcars_samples_from_labels(data_root)
            train_samples, val_samples = _stratified_split(
                full_samples,
                train_ratio=train_ratio,
                seed=seed,
            )
    else:
        mode = "auto-split"
        full_samples = _collect_samples(data_root)
        train_samples, val_samples = _stratified_split(
            full_samples,
            train_ratio=train_ratio,
            seed=seed,
        )

    train_dataset = ImageListDataset(train_samples, transform=build_train_transform(input_size))
    val_dataset = ImageListDataset(val_samples, transform=build_eval_transform(input_size))

    metadata = {
        "mode": mode,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "train_class_counts": _class_counts(train_samples),
        "val_class_counts": _class_counts(val_samples),
    }
    return train_dataset, val_dataset, metadata


def make_dataloaders(
    data_root: Path,
    input_size: int,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    seed: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, dict[str, object]]:
    train_dataset, val_dataset, metadata = build_datasets(
        data_root=data_root,
        input_size=input_size,
        train_ratio=train_ratio,
        seed=seed,
    )

    use_persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent_workers,
    )
    return train_loader, val_loader, metadata
