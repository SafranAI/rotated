import pytest
import torch

from rotated.backbones.timm import TIMM_AVAILABLE, TimmBackbone


@pytest.mark.skipif(not TIMM_AVAILABLE, reason="timm not available")
def test_timm_backbone_basic():
    """Test TimmBackbone functionality."""
    model = TimmBackbone(
        model_name="resnet18",
        pretrained=False,
        return_levels=(2, 3, 4)
    )

    input_tensor = torch.randn(1, 3, 224, 224)
    outputs = model(input_tensor)

    # Should return 3 features
    assert len(outputs) == 3

    for output in outputs:
        assert not torch.isnan(output).any()
