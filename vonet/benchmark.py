from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from vonet.constants import CLASS_NAMES
from vonet.model import VoNet
from vonet.utils import count_parameters, estimate_model_size_mb, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark VoNet inference latency and throughput."
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Path to trained checkpoint.")
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, mps, or cpu")
    parser.add_argument("--input-size", type=int, default=227)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use float16 autocast on CUDA during benchmarking.",
    )
    return parser.parse_args()


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unexpected checkpoint format in {path}")


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _benchmark_non_cuda(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
    warmup: int,
    runs: int,
) -> list[float]:
    latencies_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(inputs)
        _synchronize_device(device)

        for _ in range(runs):
            start = time.perf_counter()
            _ = model(inputs)
            _synchronize_device(device)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000.0)

    return latencies_ms


def _benchmark_cuda(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    warmup: int,
    runs: int,
    use_amp: bool,
) -> list[float]:
    latencies_ms: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _ = model(inputs)

        _synchronize_device(torch.device("cuda"))

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            start_event.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _ = model(inputs)
            end_event.record()
            _synchronize_device(torch.device("cuda"))
            latencies_ms.append(start_event.elapsed_time(end_event))

    return latencies_ms


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model = VoNet(num_classes=len(CLASS_NAMES))
    if args.checkpoint is not None:
        state_dict = _load_state_dict(args.checkpoint)
        model.load_state_dict(state_dict)

    model.eval().to(device)

    use_amp = args.amp and device.type == "cuda"
    inputs = torch.randn(args.batch_size, 3, args.input_size, args.input_size, device=device)

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint if args.checkpoint else 'none (random init)'}")
    print(f"batch_size: {args.batch_size}")
    print(f"input: {args.input_size}x{args.input_size}")
    print(f"warmup: {args.warmup}")
    print(f"runs: {args.runs}")
    print(f"amp: {use_amp}")
    print(f"parameters: {count_parameters(model):,}")
    print(f"estimated size (fp32): {estimate_model_size_mb(model):.2f} MB")

    if device.type == "cuda":
        latencies_ms = _benchmark_cuda(
            model=model,
            inputs=inputs,
            warmup=args.warmup,
            runs=args.runs,
            use_amp=use_amp,
        )
    else:
        latencies_ms = _benchmark_non_cuda(
            model=model,
            inputs=inputs,
            device=device,
            warmup=args.warmup,
            runs=args.runs,
        )

    latency = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(latency.mean())
    std_ms = float(latency.std())
    p50_ms = float(np.percentile(latency, 50))
    p90_ms = float(np.percentile(latency, 90))
    p95_ms = float(np.percentile(latency, 95))
    p99_ms = float(np.percentile(latency, 99))
    throughput_fps = args.batch_size * 1000.0 / mean_ms

    print("\nbenchmark results")
    print(f"mean latency: {mean_ms:.3f} ms")
    print(f"std latency: {std_ms:.3f} ms")
    print(f"p50 latency: {p50_ms:.3f} ms")
    print(f"p90 latency: {p90_ms:.3f} ms")
    print(f"p95 latency: {p95_ms:.3f} ms")
    print(f"p99 latency: {p99_ms:.3f} ms")
    print(f"throughput: {throughput_fps:.2f} images/s")


if __name__ == "__main__":
    main()
