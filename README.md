# Rotated Object Detection

PyTorch implementation of rotated object detection models with oriented bounding boxes.

## Features

- Rotated bounding boxes with 5-parameter format `[cx, cy, w, h, angle]`
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

model = create_ppyoloer_model(num_classes=15)
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

## Model Components

- **Backbone**: CSPResNet or TIMM models for feature extraction
- **Neck**: Custom CSP-PAN for multi-scale feature fusion
- **Head**: PP-YOLOE-R head with angle prediction
- **Post-processor**: NMS and score filtering for inference

## Acknowledgments

This implementation is adapted from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

## License

Apache License 2.0
