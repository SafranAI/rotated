import torch

from rotated.nn.ppyoloer_head import PPYOLOERHead


def test_ppyoloer_head_forward_inference():
    """Test PPYOLOERHead forward pass in inference mode."""
    in_channels = (192, 384, 768)
    num_classes = 15
    fpn_strides = (8, 16, 32)
    batch_size = 2
    img_size = 640

    head = PPYOLOERHead(in_channels=in_channels, num_classes=num_classes, fpn_strides=fpn_strides)

    # Create test features
    test_features = [
        torch.randn(batch_size, 192, img_size // 8, img_size // 8),
        torch.randn(batch_size, 384, img_size // 16, img_size // 16),
        torch.randn(batch_size, 768, img_size // 32, img_size // 32),
    ]

    head.eval()
    with torch.no_grad():
        losses, cls_scores, decoded_boxes = head(test_features)

    # Verify inference outputs
    torch.testing.assert_close(losses.total, torch.tensor(0.0))

    expected_anchors = sum((img_size // stride) ** 2 for stride in fpn_strides)
    assert cls_scores.shape == (batch_size, expected_anchors, num_classes)
    assert decoded_boxes.shape == (batch_size, expected_anchors, 5)

    assert not torch.isnan(cls_scores).any()
    assert not torch.isnan(decoded_boxes).any()
    assert torch.all(cls_scores >= 0) and torch.all(cls_scores <= 1)
