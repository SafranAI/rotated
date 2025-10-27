# Rotated Object Detection

PyTorch implementation of rotated object detection models with oriented bounding boxes.

## Features

- Rotated bounding boxes with 5-parameter format `[cx, cy, w, h, angle]`
- Probabilistic IoU for differentiable loss computation
- Task-aligned assignment for dynamic anchor matching
- Modular architecture with backbone, neck, and head components

## Quick Start

```python
from rotated.architecture import create_ppyoloer_model

model = create_ppyoloer_model(num_classes=15)
images = torch.randn(2, 3, 640, 640)

# Training
targets = {
    'labels': torch.randint(0, 15, (2, 5, 1)),
    'boxes': torch.rand(2, 5, 5) * 100,
    'valid_mask': torch.ones(2, 5, 1)
}
losses, boxes, socres, labels = model(images, targets)

# Inference
_, boxes, socres, labels = model(images)
```

## Acknowledgments

This implementation is adapted from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection).

## License

Apache License 2.0
