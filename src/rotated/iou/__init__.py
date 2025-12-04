import inspect
from typing import Any, Literal, TypeAlias

from rotated.iou.approx_iou import ApproxRotatedIoU
from rotated.iou.approx_sdf import ApproxSDFL1
from rotated.iou.precise_iou import PreciseRotatedIoU
from rotated.iou.prob_iou import ProbIoU

__all__ = ["ApproxRotatedIoU", "ApproxSDFL1", "PreciseRotatedIoU", "ProbIoU", "iou_picker"]


_IOU_METHODS = {
    "approx_sdf_l1": ApproxSDFL1,
    "precise_rotated_iou": PreciseRotatedIoU,
    "prob_iou": ProbIoU,
    "approx_rotated_iou": ApproxRotatedIoU,
}

IoUMethod: TypeAlias = ApproxRotatedIoU | ApproxSDFL1 | PreciseRotatedIoU | ProbIoU
IoUMethodName: TypeAlias = Literal["approx_rotated_iou", "approx_sdf_l1", "precise_rotated_iou", "prob_iou"]
IoUKwargs: TypeAlias = dict[str, Any] | None


def iou_picker(iou_method: IoUMethodName, iou_kwargs: IoUKwargs = None) -> IoUMethod:
    """Validate and instantiate the IoU method.

    Args:
        iou_method: Method name to compute Intersection Over Union
        iou_kwargs: dictionary with parameters for the IoU method

    Returns:
        Instantiated IoU class

    Raises:
        ValueError: if unknown iou_method or unexpected IoU parameter is provided
    """
    if iou_method not in _IOU_METHODS:
        raise ValueError(f"Unknown IoU method: {iou_method}, use one of {_IOU_METHODS.keys()} instead.")

    iou_cls = _IOU_METHODS[iou_method]

    # simple instantiation if no kwargs
    if not iou_kwargs:
        return iou_cls()

    valid_params = {}
    sig = inspect.signature(iou_cls.__init__)

    expected_params = set(sig.parameters.keys()) - {"self"}

    for param_name, param_value in iou_kwargs.items():
        if param_name in expected_params:
            valid_params[param_name] = param_value
        else:
            raise ValueError(
                f"Parameter '{param_name}' is not supported by {iou_cls.__name__}."
                f" Supported parameters: {', '.join(expected_params)}"
            )

    return iou_cls(**valid_params)
