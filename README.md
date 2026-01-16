# Rotated Object Detection

PyTorch implementation of rotated object detection models with oriented bounding boxes.

## Features

- Rotated bounding boxes with 5-parameter format `[cx, cy, w, h, angle]`
- Pretrained CSPResNet backbones (s/m/l/x variants)
- Probabilistic IoU for differentiable loss computation
- Task-aligned assignment for dynamic anchor matching
- Modular architecture with backbone, neck, and head components

## Box Format

All bounding boxes use the format `[cx, cy, w, h, angle]` where:

- `cx, cy`: Center coordinates in absolute pixels (relative to input image dimensions)
- `w, h`: Width and height in absolute pixels
- `angle`: Rotation angle in **radians**, range **[0, π/2)** (equivalent to 0-90 degrees in OpenCV format)

The angle range [0, π/2) follows the OpenCV convention where angles are measured from the horizontal axis.

## Quick Start

```python
import math
import torch
from rotated.models import create_ppyoloer_model

# Create model with pretrained backbone (default)
model = create_ppyoloer_model(num_classes=15, size="s")

# Or without pretrained backbone
model = create_ppyoloer_model(num_classes=15, size="s", pretrained_backbone=False)

images = torch.randn(2, 3, 640, 640)

# Training
targets = {
    'labels': torch.randint(0, 15, (2, 5, 1)),
    'boxes': torch.cat([
        torch.rand(2, 5, 4) * 100,  # cx, cy, w, h
        torch.rand(2, 5, 1) * (math.pi / 2)  # angle in [0, π/2)
    ], dim=-1),
    'valid_mask': torch.ones(2, 5, 1)
}
losses, boxes, scores, labels = model(images, targets)

# Inference
_, boxes, scores, labels = model(images)
```

## Model Variants

Available model sizes (s/m/l/x) with different capacity/accuracy trade-offs:

- **s** (small): Fastest, lowest memory
- **m** (medium): Balanced speed and accuracy
- **l** (large): Higher accuracy
- **x** (extra large): Best accuracy, highest compute

## Using Pretrained Backbones

The CSPResNet backbone can be used directly with pretrained weights:

```python
from rotated.backbones import create_csp_resnet

# Create backbone with pretrained weights
backbone = create_csp_resnet("s", pretrained=True)

# Or without pretrained weights
backbone = create_csp_resnet("l", pretrained=False)
```

Pretrained weights are automatically downloaded and cached on first use.

## Model Components

- **Backbone**: CSPResNet (with pretrained weights) or TIMM models for feature extraction
- **Neck**: Custom CSP-PAN for multi-scale feature fusion
- **Head**: PP-YOLOE-R head with angle prediction
- **Post-processor**: NMS and score filtering for inference

## Acknowledgments

This implementation is adapted from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

## License

Apache License 2.0