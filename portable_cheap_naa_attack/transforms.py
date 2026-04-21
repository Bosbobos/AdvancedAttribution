from __future__ import annotations

"""Input and gradient transformations used by the PD variant."""

import math

import torch
import torch.nn.functional as F


def apply_dim(
    x: torch.Tensor,
    *,
    probability: float,
    resize_low: float = 1.0,
    resize_high: float = 1.1,
) -> torch.Tensor:
    """Apply a differentiable DIM-style random resize-and-pad transform."""

    if probability <= 0.0:
        return x
    if torch.rand((), device=x.device).item() > probability:
        return x

    height, width = x.shape[-2:]
    base = max(height, width)
    min_resize = max(1, int(math.floor(base * resize_low)))
    max_resize = max(base, int(math.ceil(base * resize_high)))
    resized_side = int(torch.randint(max(base, min_resize), max_resize + 1, (1,), device=x.device).item())
    if resized_side == base and max_resize == base:
        return x

    resized = F.interpolate(x, size=(resized_side, resized_side), mode="bilinear", align_corners=False)
    top = int(torch.randint(0, max_resize - resized_side + 1, (1,), device=x.device).item())
    left = int(torch.randint(0, max_resize - resized_side + 1, (1,), device=x.device).item())
    bottom = max_resize - resized_side - top
    right = max_resize - resized_side - left

    padded = F.pad(resized, (left, right, top, bottom), mode="constant", value=0.0)
    if padded.shape[-2:] == (height, width):
        return padded
    return F.interpolate(padded, size=(height, width), mode="bilinear", align_corners=False)


def apply_pim_to_gradient(
    grad: torch.Tensor,
    *,
    amplification_factor: float,
    kernel_size: int,
) -> torch.Tensor:
    """A compact PIM-style patch-aware smoothing and amplification of the gradient."""

    if amplification_factor == 1.0 and kernel_size == 1:
        return grad

    channels = grad.shape[1]
    kernel = torch.ones(
        channels,
        1,
        kernel_size,
        kernel_size,
        device=grad.device,
        dtype=grad.dtype,
    )
    kernel = kernel / float(kernel_size * kernel_size)
    smoothed = F.conv2d(
        grad,
        weight=kernel,
        bias=None,
        stride=1,
        padding=kernel_size // 2,
        groups=channels,
    )
    return amplification_factor * smoothed
