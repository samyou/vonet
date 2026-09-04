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

This repository contains a PyTorch reproduction of **VoNet** for five-class vehicle-orientation classification. The classes are `front`, `rear`, `side`, `front-side`, and `rear-side`.

The checkpoint selected by validation accuracy is from epoch **28** of a 30-epoch run:

- Validation accuracy: **94.5378%**
- Validation loss: **0.1774**
- Training accuracy at the selected epoch: **94.0809%**
- Parameters: **396,949**
- Input resolution: **227 x 227 RGB**

This revised implementation follows the paper's module-level design, but the paper did not publish every layer width, so results and parameter count differ slightly from the original Caffe implementation.

## Usage

Loading this custom architecture executes the Python files in this repository. Review them before setting `trust_remote_code=True`.

```python
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

repo_id = "samyou/vonet-compcars"
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
- Epochs: 30
- Learning rate: 0.001
- Momentum: 0.9
- Weight decay: 0.0005
- Dropout: 0.5
- Initialization: Xavier uniform
- Augmentation: random horizontal flip
- Normalization: ImageNet mean and standard deviation

Detailed checkpoint metrics and arguments are in `training_summary.json`.

## Intended use and limitations

The model predicts coarse vehicle orientation from a single RGB image. It does not distinguish left from right and may be unreliable under occlusion, unusual viewpoints, image-domain shift, or for vehicle designs poorly represented by CompCars. Validate it for your environment before safety-critical use. CompCars dataset terms remain applicable to uses involving the original dataset.

## References

VoNet:

```bibtex
@inproceedings{you2016vonet,
  title={VoNet: Vehicle Orientation Classification Using Convolutional Neural Network},
  author={You, Ratanaksamrith and Kwon, Jang-Woo},
  booktitle={Proceedings of the 2nd International Conference on Communication and Information Processing},
  year={2016},
  doi={10.1145/3018009.3018045}
}
```

CompCars:

```bibtex
@inproceedings{yang2015compcars,
  title={A Large-Scale Car Dataset for Fine-Grained Categorization and Verification},
  author={Yang, Linjie and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2015}
}
```

Source implementation: https://github.com/samyou/vonet
