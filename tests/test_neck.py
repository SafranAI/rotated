import torch

from rotated.neck import CustomCSPPAN


def test_custom_csp_pan():
    """Test CustomCSPPAN functionality."""
    neck = CustomCSPPAN(
        in_channels=(256, 512, 1024),
        out_channels=(192, 384, 768),
        act="swish",
    )

    test_features = [
        torch.randn(2, 256, 80, 80),   # P3: stride 8
        torch.randn(2, 512, 40, 40),   # P4: stride 16
        torch.randn(2, 1024, 20, 20),  # P5: stride 32
    ]

    outputs = neck(test_features)

    expected_channels = (192, 384, 768)
    assert len(outputs) == 3
    for i, output in enumerate(outputs):
        assert output.shape[1] == expected_channels[i]
        assert not torch.isnan(output).any()
