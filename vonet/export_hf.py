from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from textwrap import dedent
from typing import Any

import torch
from safetensors.torch import save_file

import vonet.model as model_module
from vonet.constants import CLASS_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained VoNet checkpoint as a Hugging Face model package."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vonet/huggingface"),
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Hugging Face model ID used in the generated model card.",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default="https://github.com/samyou/vonet",
    )
    parser.add_argument("--license-file", type=Path, default=Path("LICENSE"))
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict[str, Any]:
    raw_checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw_checkpoint, dict):
        raise ValueError(f"Expected a checkpoint dictionary in {path}.")

    if "model_state" in raw_checkpoint:
        checkpoint = raw_checkpoint
    elif raw_checkpoint and all(
        isinstance(value, torch.Tensor) for value in raw_checkpoint.values()
    ):
        checkpoint = {"model_state": raw_checkpoint}
    else:
        raise ValueError(
            f"{path} does not contain 'model_state' or a raw tensor state dictionary."
        )

    state_dict = checkpoint["model_state"]
    if not isinstance(state_dict, dict) or not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise ValueError(f"Invalid model state dictionary in {path}.")
    return checkpoint


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _render_configuration_code() -> str:
    return dedent(
        """\
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
        """
    )


def _render_modeling_code() -> str:
    source = Path(inspect.getfile(model_module)).read_text(encoding="utf-8")
    local_imports = (
        "import torch\n"
        "from torch import nn\n\n"
        "from vonet.utils import xavier_init"
    )
    if local_imports not in source:
        raise RuntimeError("Could not locate the expected imports in vonet.model.")

    hub_imports_and_initializer = dedent(
        """\
        from typing import Optional, Tuple, Union

        import torch
        from torch import nn
        from transformers import PreTrainedModel
        from transformers.modeling_outputs import ImageClassifierOutput

        from .configuration_vonet import VoNetConfig


        def xavier_init(module: nn.Module) -> None:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        """
    ).rstrip()
    source = source.replace(local_imports, hub_imports_and_initializer)

    wrapper = dedent(
        """

        class VoNetForImageClassification(PreTrainedModel):
            config_class = VoNetConfig
            main_input_name = "pixel_values"

            def __init__(self, config: VoNetConfig) -> None:
                super().__init__(config)
                self.vonet = VoNet(
                    num_classes=config.num_labels,
                    dropout=config.dropout,
                )
                self.post_init()

            def forward(
                self,
                pixel_values: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = None,
            ) -> Union[ImageClassifierOutput, Tuple[torch.Tensor, ...]]:
                if return_dict is None:
                    return_dict = getattr(self.config, "return_dict", True)

                logits = self.vonet(pixel_values)
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(logits, labels)

                if not return_dict:
                    output = (logits,)
                    return ((loss,) + output) if loss is not None else output
                return ImageClassifierOutput(loss=loss, logits=logits)
        """
    )
    return source.rstrip() + wrapper


def _render_model_card(
    repo_id: str,
    checkpoint: dict[str, Any],
    class_names: list[str],
    parameter_count: int,
    input_size: int,
    source_url: str,
) -> str:
    epoch = checkpoint.get("epoch")
    training_args = checkpoint.get("args", {})
    total_epochs = training_args.get("epochs") if isinstance(training_args, dict) else None
    epoch_text = (
        f"epoch **{epoch}** of a {total_epochs}-epoch run"
        if epoch is not None and total_epochs is not None
        else "the supplied checkpoint"
    )

    metric_lines: list[str] = []
    if "val_acc" in checkpoint:
        metric_lines.append(f"- Validation accuracy: **{float(checkpoint['val_acc']):.4%}**")
    if "val_loss" in checkpoint:
        metric_lines.append(f"- Validation loss: **{float(checkpoint['val_loss']):.4f}**")
    if "train_acc" in checkpoint:
        metric_lines.append(
            f"- Training accuracy at the selected epoch: "
            f"**{float(checkpoint['train_acc']):.4%}**"
        )
    metric_lines.extend(
        [
            f"- Parameters: **{parameter_count:,}**",
            f"- Input resolution: **{input_size} x {input_size} RGB**",
        ]
    )
    metrics = "\n".join(metric_lines)
    labels = ", ".join(f"`{class_name}`" for class_name in class_names)

    return dedent(
        f"""\
        ---
        license: mit
        library_name: transformers
        pipeline_tag: image-classification
        metrics:
        - accuracy
        tags:
        - pytorch
        - vehicle-orientation
        - compcars
        - custom-code
        ---

        # VoNet for CompCars vehicle orientation

        This repository contains a PyTorch reproduction of **VoNet** for
        five-class vehicle-orientation classification. The classes are {labels}.

        The exported model uses {epoch_text}:

        {metrics}

        This revised implementation follows the paper's module-level design, but
        the paper did not publish every layer width, so results and parameter
        count differ slightly from the original Caffe implementation.

        ## Usage

        Loading this custom architecture executes the Python files in this
        repository. Review them before setting `trust_remote_code=True`.

        ```python
        from PIL import Image
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        repo_id = "{repo_id}"
        processor = AutoImageProcessor.from_pretrained(repo_id)
        model = AutoModelForImageClassification.from_pretrained(
            repo_id,
            trust_remote_code=True,
        ).eval()

        image = Image.open("car.jpg").convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            logits = model(**inputs).logits

        predicted_id = logits.argmax(dim=-1).item()
        print(model.config.id2label[predicted_id])
        ```

        ## Training

        - Dataset: CompCars vehicle viewpoint annotations
        - Optimizer: SGD
        - Epochs: {total_epochs if total_epochs is not None else "see training_summary.json"}
        - Learning rate: {training_args.get("lr", "see training_summary.json") if isinstance(training_args, dict) else "see training_summary.json"}
        - Momentum: {training_args.get("momentum", "see training_summary.json") if isinstance(training_args, dict) else "see training_summary.json"}
        - Weight decay: {training_args.get("weight_decay", "see training_summary.json") if isinstance(training_args, dict) else "see training_summary.json"}
        - Dropout: {training_args.get("dropout", "see training_summary.json") if isinstance(training_args, dict) else "see training_summary.json"}
        - Initialization: Xavier uniform
        - Augmentation: random horizontal flip
        - Normalization: ImageNet mean and standard deviation

        Detailed checkpoint metrics and arguments are in `training_summary.json`.

        ## Intended use and limitations

        The model predicts coarse vehicle orientation from a single RGB image.
        It does not distinguish left from right and may be unreliable under
        occlusion, unusual viewpoints, image-domain shift, or for vehicle designs
        poorly represented by CompCars. Validate it for your environment before
        safety-critical use. CompCars dataset terms remain applicable to uses
        involving the original dataset.

        ## References

        VoNet:

        ```bibtex
        @inproceedings{{you2016vonet,
          title={{VoNet: Vehicle Orientation Classification Using Convolutional Neural Network}},
          author={{You, Ratanaksamrith and Kwon, Jang-Woo}},
          booktitle={{Proceedings of the 2nd International Conference on Communication and Information Processing}},
          year={{2016}},
          doi={{10.1145/3018009.3018045}}
        }}
        ```

        CompCars:

        ```bibtex
        @inproceedings{{yang2015compcars,
          title={{A Large-Scale Car Dataset for Fine-Grained Categorization and Verification}},
          author={{Yang, Linjie and Luo, Ping and Loy, Chen Change and Tang, Xiaoou}},
          booktitle={{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}},
          year={{2015}}
        }}
        ```

        Source implementation: {source_url}
        """
    )


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    repo_id: str,
    source_url: str,
    license_file: Path,
) -> None:
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict = checkpoint["model_state"]
    class_names = list(checkpoint.get("class_names", CLASS_NAMES))
    training_args = checkpoint.get("args", {})
    dropout = (
        float(training_args.get("dropout", 0.5))
        if isinstance(training_args, dict)
        else 0.5
    )
    input_size = (
        int(training_args.get("input_size", 227))
        if isinstance(training_args, dict)
        else 227
    )

    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is not None and classifier_weight.shape[0] != len(class_names):
        raise ValueError(
            "The checkpoint classifier output count does not match its class names."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    hub_state_dict = {
        (key if key.startswith("vonet.") else f"vonet.{key}"): value.detach()
        .cpu()
        .contiguous()
        for key, value in state_dict.items()
    }
    save_file(
        hub_state_dict,
        output_dir / "model.safetensors",
        metadata={"format": "pt"},
    )

    id2label = {str(index): label for index, label in enumerate(class_names)}
    label2id = {label: index for index, label in enumerate(class_names)}
    _write_json(
        output_dir / "config.json",
        {
            "architectures": ["VoNetForImageClassification"],
            "auto_map": {
                "AutoConfig": "configuration_vonet.VoNetConfig",
                "AutoModelForImageClassification": (
                    "modeling_vonet.VoNetForImageClassification"
                ),
            },
            "dropout": dropout,
            "id2label": id2label,
            "input_size": input_size,
            "label2id": label2id,
            "model_type": "vonet",
            "num_channels": 3,
            "num_labels": len(class_names),
            "torch_dtype": "float32",
        },
    )
    _write_json(
        output_dir / "preprocessor_config.json",
        {
            "do_center_crop": False,
            "do_convert_rgb": True,
            "do_normalize": True,
            "do_rescale": True,
            "do_resize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_processor_type": "ViTImageProcessor",
            "image_std": [0.229, 0.224, 0.225],
            "resample": 2,
            "rescale_factor": 1 / 255,
            "size": {"height": input_size, "width": input_size},
        },
    )

    summary = {
        "source_checkpoint": str(checkpoint_path),
        "selected_epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "train_loss": checkpoint.get("train_loss"),
        "train_accuracy": checkpoint.get("train_acc"),
        "validation_loss": checkpoint.get("val_loss"),
        "validation_accuracy": checkpoint.get("val_acc"),
        "best_validation_accuracy": checkpoint.get("best_val_acc"),
        "training_arguments": training_args,
    }
    _write_json(output_dir / "training_summary.json", summary)

    parameter_count = sum(tensor.numel() for tensor in state_dict.values())
    (output_dir / "configuration_vonet.py").write_text(
        _render_configuration_code(),
        encoding="utf-8",
    )
    (output_dir / "modeling_vonet.py").write_text(
        _render_modeling_code(),
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        _render_model_card(
            repo_id=repo_id,
            checkpoint=checkpoint,
            class_names=class_names,
            parameter_count=parameter_count,
            input_size=input_size,
            source_url=source_url,
        ),
        encoding="utf-8",
    )
    if license_file.exists():
        (output_dir / "LICENSE").write_text(
            license_file.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    print(f"exported Hugging Face package: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size:,} bytes")


def main() -> None:
    args = parse_args()
    export_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        source_url=args.source_url,
        license_file=args.license_file,
    )


if __name__ == "__main__":
    main()
