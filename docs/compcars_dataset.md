# CompCars Dataset Documentation

This document provides an overview and structure description of the
**CompCars** dataset.\
Official website:
http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html


## Citation

If you use the CompCars dataset in your research, please cite the
following paper:

Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang.\
**"A Large-Scale Car Dataset for Fine-Grained Categorization and
Verification"**\
In *Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition (CVPR)*, 2015.


## Directory Structure

```text
CompCars/
├── image/
├── label/
├── misc/
├── part/
└── train_test_split/
```


## 1. image/

Stores all full car images using the following path format:

make_id/model_id/released_year/image_name.jpg


## 2. label/

Stores label files corresponding to full car images using the same path
structure:

make_id/model_id/released_year/image_name.txt

Each label file contains three lines:

1.  Viewpoint annotation
    -   -1 --- Uncertain
    -   1 --- Front
    -   2 --- Rear
    -   3 --- Side
    -   4 --- Front-side
    -   5 --- Rear-side
2.  Number of bounding boxes
    -   Always 1 in the current release.
3.  Bounding box coordinates

x1 y1 x2 y2

Constraints:
- 1 ≤ x1 < x2 ≤ image_width
- 1 ≤ y1 < y2 ≤ image_height


## 3. misc/

### attributes.txt

Each line corresponds to one car model:

model_id maximum_speed displacement door_number seat_number type

-   type ranges from 1-12
-   Type mappings are defined in car_type.mat
-   Unavailable attributes are denoted by 0 or 0.0

### make_model_name.mat

Contains:
- make_names --- maps make_id to make names
- model_names --- maps model_id to model names


## 4. part/

Stores car part images:

make_id/model_id/released_year/part_id/image_name.jpg

### Part ID Mapping

1 --- Headlight\
2 --- Taillight\
3 --- Fog light\
4 --- Air intake\
5 --- Console\
6 --- Steering wheel\
7 --- Dashboard\
8 --- Gear lever


## 5. train_test_split/

Provides the train/test subsets used in the CVPR 2015 paper.

### classification/

Train/test lists for full car image classification.

### part/

Train/test lists for car part classification.

### verification/

Files included:

-   verification_train.txt
    -   Image list for training verification models
    -   Also used for testing attribute prediction
-   verification_pairs_easy.txt
-   verification_pairs_medium.txt
-   verification_pairs_hard.txt

Each line in verification_pairs_XXX.txt:

path_to_image_1 path_to_image_2 label

Where:
- label = 1 → Positive pair
- label = 0 → Negative pair



## Summary

The CompCars dataset provides:

-   Full car images
-   Part-level images
-   Viewpoint annotations
-   Bounding box annotations
-   Attribute labels
-   Verification pairs
-   Standard train/test splits

Designed for:

-   Fine-grained car classification
-   Attribute prediction
-   Car verification
-   Part-based recognition
