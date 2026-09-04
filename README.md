# VoNet

Reproducible implementation of the VoNet paper (`ICCIP 2016`) for 5-class vehicle orientation classification:
`front`, `rear`, `side`, `front-side`, `rear-side`.

Original implementation in the paper used Caffe framework. This is a revised implementation in PyTorch.
The result can be slightly differ from what stated in the paper.

## Install

```bash
uv sync
```

Run commands without activating a venv:

```bash
uv run vonet-train --data-root data --device auto
uv run vonet-infer --checkpoint artifacts/vonet/best.pt --image data/image/1/1101/2011/07b90decb92ba6.jpg --device auto
uv run vonet-benchmark --checkpoint artifacts/vonet/best.pt --device auto
```

## Dataset Layout

VoNet now supports three dataset layouts.

1) Pre-split class folders:

```text
data/
  train/front ...
  train/rear ...
  train/side ...
  train/front-side ...
  train/rear-side ...
  val/front ...
  val/rear ...
  val/side ...
  val/front-side ...
  val/rear-side ...
```

2) Single folder (`data/front`, `data/rear`, ...) and the trainer auto-splits with `--train-ratio`.

3) Raw CompCars root:

```text
CompCars/
  image/make_id/model_id/released_year/image_name.jpg
  label/make_id/model_id/released_year/image_name.txt
  train_test_split/classification/*.txt  # optional
```

Accepted class directory names:
- `front`, `rear`, `side`
- `front-side` (also `front_side`, `frontside`)
- `rear-side` (also `rear_side`, `rearside`)

For raw CompCars, class labels are read from the first line in each file under `label/`:
- `1 -> front`, `2 -> rear`, `3 -> side`, `4 -> front-side`, `5 -> rear-side`
- `-1` (uncertain) is skipped
- if `train_test_split/classification` train/test files exist, they are used
- otherwise the loader builds a stratified split using `--train-ratio`
- raw dataset structure reference: `docs/compcars_dataset.md`

## Train

```bash
uv run vonet-train \
  --data-root data \
  --epochs 30 \
  --lr 0.001 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --dropout 0.5 \
  --input-size 227 \
  --device auto
```

Resume from the last checkpoint:

```bash
uv run vonet-train \
  --data-root data \
  --epochs 30 \
  --resume artifacts/vonet/last.pt \
  --device auto
```

Notes:
- `--resume` restores model/optimizer state and continues from the next epoch.
- Keep `--epochs` as the final target epoch number (for example, resume from epoch 25 with `--epochs 30` to run 26-30).
- Training logs include `epoch_time` per epoch and total `duration` at the end.

Apple Silicon (M4/MPS):

```bash
uv run vonet-train \
  --data-root data \
  --device mps \
  --num-workers 0
```

## Export to Hugging Face

Generate a Transformers-compatible model package from the best checkpoint:

```bash
uv run vonet-export-hf \
  --checkpoint artifacts/vonet/best.pt \
  --output-dir artifacts/vonet/huggingface \
  --repo-id samyou/vonet-compcars
```

The export contains safetensors weights, model and preprocessing
configuration, custom Transformers model code, training metadata, and a model
card. Authenticate and upload the generated directory with:

```bash
hf auth login
hf repos create samyou/vonet-compcars --type model --public --exist-ok
hf upload samyou/vonet-compcars artifacts/vonet/huggingface .
```

## Inference

```bash
uv run vonet-infer --checkpoint artifacts/vonet/best.pt --image sample.jpg --device auto
```

## Benchmark Inference

```bash
uv run vonet-benchmark \
  --checkpoint artifacts/vonet/best.pt \
  --device auto \
  --batch-size 1 \
  --warmup 50 \
  --runs 300
```

Apple Silicon:

```bash
uv run vonet-benchmark --checkpoint artifacts/vonet/best.pt --device mps --runs 500
```

## License

MIT. See `LICENSE`.
