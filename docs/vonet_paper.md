# VoNet: Vehicle Orientation Classification Using Convolutional Neural Network

**Authors:**
Ratanaksamrith You, Jang-Woo Kwon
Department of Computer and Information Engineering
Inha University, South Korea

**Conference:** ICCIP 2016
**DOI:** [https://doi.org/10.1145/3018009.3018045](https://doi.org/10.1145/3018009.3018045)

---

## Abstract

This paper presents a convolutional neural network (CNN) for classifying vehicle orientation (viewpoint) from a single image.

Target classes:

1. Front
2. Rear
3. Side
4. Front-side
5. Rear-side

Key results:

* **Accuracy:** ~95%
* **Inference time:** 57 ms (NVIDIA GRID K520 GPU)
* **Model size:** 1.6 MB
* **Framework:** Caffe

The goal is to create a lightweight, fast, and accurate model suitable for autonomous driving systems.

---

## 1. Introduction

Current autonomous vehicle systems can detect vehicles using bounding boxes but **do not identify orientation**.

Two ways to determine vehicle heading:

1. Track motion over time (requires temporal data)
2. Infer orientation from a single image (this paper’s focus)

Orientation is critical in complex urban environments where motion alone may be misleading (e.g., reversing vs accelerating toward sensor).

Design goals:

* Simplicity
* Speed
* Accuracy
* Lightweight model for embedded systems

---

## 2. Related Work

Relevant CNN architectures:

* LeNet
* AlexNet
* GoogLeNet
* VGGNet
* ResNet
* SqueezeNet

Model compression techniques:

* Low-rank approximation
* Hashing
* Quantization
* Pruning

No prior research specifically addressed vehicle viewpoint classification using CNNs.

---

## 3. Dataset

### CompCars Dataset

* 208,826 total images
* 1,716 car models
* Web-nature and surveillance-nature images
* URL: https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars

Used subset:

* 16,016 full-car images
* Divided into:

| Class      | Images |
| ---------- | ------ |
| Front      | 2,593  |
| Rear       | 1,997  |
| Side       | 2,973  |
| Front-side | 4,828  |
| Rear-side  | 3,625  |

Train set: 12,013 images
Validation set: 4,003 images

---

## 4. Network Architecture: VoNet

VoNet is inspired by:

* **GoogLeNet (Inception module)**
* **SqueezeNet (Fire module)**

### VoNet Modules

#### I-Module

* Based on Inception
* Removes 1×1 and 5×5 convolutions
* Uses 3×3 convolutions

#### F-Module

* Based on Fire module
* Removes e1×1 layer
* Uses 1×1 and 3×3 convolutions

### Architectural Principles

* Use larger receptive fields early
* Use 1×1 convolutions later for dimensionality reduction
* Remove fully-connected layers
* Reduce parameter count

---

## 5. Training Details

Framework: Caffe
Optimizer: SGD
Weight initialization: Xavier
Weight decay: 0.0005
Momentum: 0.9
Dropout: 0.5

### Input Sizes

* AlexNet / SqueezeNet / VoNet: 227×227
* GoogLeNet: 224×224

### Training Approaches

* From scratch
* Partial fine-tuning
* Full fine-tuning

VoNet training:

* Learning rate: 0.001
* 30 epochs

---

## 6. Experimental Results

Hardware:

* NVIDIA GRID K520 GPU (4GB)

### Performance Comparison

| Model                  | Accuracy   | Inference (ms) | Size (MB) | Parameters  |
| ---------------------- | ---------- | -------------- | --------- | ----------- |
| AlexNet                | 0.9714     | 179.23         | 227       | 56,888,709  |
| AlexNet (Partial FT)   | 0.9509     | 314.52         | —         | —           |
| GoogLeNet              | 0.9732     | 106.50         | 41.3      | 10,318,655  |
| GoogLeNet (Full FT)    | **0.9770** | 106.47         | —         | —           |
| GoogLeNet (Partial FT) | 0.9588     | 166.11         | —         | —           |
| SqueezeNet             | 0.9614     | 65.92          | 2.9       | 725,061     |
| **VoNet**              | 0.9545     | **57.04**      | **1.6**   | **394,789** |

### Observations

* Best accuracy: GoogLeNet (Full fine-tune)
* Fastest inference: VoNet
* Smallest model: VoNet
* Fewest parameters: VoNet

Tradeoff:
VoNet sacrifices ~2% accuracy for significant improvements in speed and size.

---

## 7. Discussion

Limitations:

* Cannot distinguish left vs right side due to dataset annotation
* Dataset angle distribution affects accuracy
* Modern car design sometimes confuses predictions

Improvements:

* Add left/right orientation labels
* Uniform data distribution
* Fuse with lidar/radar for hybrid prediction

Real-time constraint:

* Video frame budget ≈ 33ms (30 fps)
* VoNet at 57ms is acceptable since orientation changes slowly (1–2 seconds)

---

## 8. Conclusion

The paper demonstrates that vehicle orientation classification can be effectively solved using CNNs without manual feature engineering.

VoNet achieves:

* ~95% accuracy
* 57 ms inference
* 1.6 MB model size
* 394k parameters

VoNet is:

* Simple
* Lightweight
* Fast
* Suitable for embedded autonomous systems

---

## 9. References

24 references including:

* AlexNet (Krizhevsky et al., 2012)
* GoogLeNet (Szegedy et al., 2015)
* SqueezeNet (Iandola et al., 2016)
* CompCars dataset (Yang et al., 2015)
* Caffe framework (Jia et al., 2014)
