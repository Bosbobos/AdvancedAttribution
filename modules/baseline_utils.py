from __future__ import annotations

"""Shared baseline helpers for path-based attribution methods."""

from typing import Mapping

import numpy as np
import torch
from PIL import Image, ImageFilter


DEFAULT_BASELINE_MODE = "zero"
DEFAULT_BLUR_SIGMA = 16.0
_SUPPORTED_BASELINE_MODES = {"zero", "mean_rgb", "blur"}


def normalize_baseline_mode(mode):
    value = str(DEFAULT_BASELINE_MODE if mode is None else mode).strip().lower()
    if value not in _SUPPORTED_BASELINE_MODES:
        raise ValueError(
            f"Unsupported baseline_mode: {mode!r}. Supported values: {sorted(_SUPPORTED_BASELINE_MODES)}"
        )
    return value


def normalize_blur_sigma(value):
    sigma = float(DEFAULT_BLUR_SIGMA if value is None else value)
    if sigma <= 0.0:
        raise ValueError(f"baseline_blur_sigma must be > 0, got {sigma}")
    return sigma


def normalize_baseline_rgb(rgb):
    if rgb is None:
        return None
    values = np.asarray(rgb, dtype=np.float32).reshape(-1)
    if values.size != 3:
        raise ValueError(f"baseline_rgb must contain exactly 3 values, got shape={tuple(values.shape)}")
    values = np.clip(values, 0.0, 1.0)
    return tuple(float(v) for v in values.tolist())


def canonicalize_baseline_config(config):
    normalized = dict(config)
    mode = normalize_baseline_mode(normalized.get("baseline_mode", DEFAULT_BASELINE_MODE))
    if mode == "zero":
        normalized.pop("baseline_mode", None)
        normalized.pop("baseline_rgb", None)
        normalized.pop("baseline_blur_sigma", None)
        return normalized

    normalized["baseline_mode"] = mode
    if mode == "mean_rgb":
        rgb = normalize_baseline_rgb(normalized.get("baseline_rgb"))
        if rgb is None:
            normalized.pop("baseline_rgb", None)
        else:
            normalized["baseline_rgb"] = [float(v) for v in rgb]
        normalized.pop("baseline_blur_sigma", None)
        return normalized

    normalized["baseline_blur_sigma"] = float(
        normalize_blur_sigma(normalized.get("baseline_blur_sigma", DEFAULT_BLUR_SIGMA))
    )
    normalized.pop("baseline_rgb", None)
    return normalized


def baseline_title_fragment(mode, *, baseline_rgb=None, blur_sigma=DEFAULT_BLUR_SIGMA):
    mode = normalize_baseline_mode(mode)
    if mode == "zero":
        return "baseline=zero"
    if mode == "mean_rgb":
        rgb = normalize_baseline_rgb(baseline_rgb)
        if rgb is None:
            return "baseline=mean_rgb"
        return "baseline=mean_rgb({})".format(",".join(f"{value:.3f}" for value in rgb))
    return f"baseline=blur,sigma={normalize_blur_sigma(blur_sigma):.3f}"


def baseline_method_fragment(mode, *, baseline_rgb=None, blur_sigma=DEFAULT_BLUR_SIGMA):
    mode = normalize_baseline_mode(mode)
    if mode == "zero":
        return ""
    if mode == "mean_rgb":
        rgb = normalize_baseline_rgb(baseline_rgb)
        if rgb is None:
            return "/baseline-mean_rgb"
        rgb_fragment = "-".join(f"{value:.2f}" for value in rgb)
        return f"/baseline-mean_rgb-{rgb_fragment}"
    return f"/baseline-blur-s{normalize_blur_sigma(blur_sigma):g}"


def baseline_display_name(mode, *, baseline_rgb=None, blur_sigma=DEFAULT_BLUR_SIGMA):
    mode = normalize_baseline_mode(mode)
    if mode == "zero":
        return "zero"
    if mode == "mean_rgb":
        rgb = normalize_baseline_rgb(baseline_rgb)
        if rgb is None:
            return "mean_rgb"
        return "mean_rgb({})".format(", ".join(f"{value:.3f}" for value in rgb))
    return f"blur(sigma={normalize_blur_sigma(blur_sigma):g})"


def build_image_baseline(
    x,
    image_np,
    *,
    mode=DEFAULT_BASELINE_MODE,
    baseline_rgb=None,
    blur_sigma=DEFAULT_BLUR_SIGMA,
):
    mode = normalize_baseline_mode(mode)
    image_np = np.asarray(image_np, dtype=np.float32)

    if mode == "zero":
        baseline_np = np.zeros_like(image_np, dtype=np.float32)
        info = {
            "baseline_mode": mode,
            "baseline_rgb": None,
            "baseline_blur_sigma": None,
        }
    elif mode == "mean_rgb":
        rgb = normalize_baseline_rgb(baseline_rgb)
        if rgb is None:
            rgb = normalize_baseline_rgb(image_np.reshape(-1, image_np.shape[-1]).mean(axis=0))
        rgb_arr = np.asarray(rgb, dtype=np.float32).reshape(1, 1, 3)
        baseline_np = np.broadcast_to(rgb_arr, image_np.shape).copy()
        info = {
            "baseline_mode": mode,
            "baseline_rgb": [float(v) for v in rgb],
            "baseline_blur_sigma": None,
        }
    else:
        sigma = normalize_blur_sigma(blur_sigma)
        pil_image = Image.fromarray(np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8))
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=sigma))
        baseline_np = np.asarray(blurred, dtype=np.float32) / 255.0
        info = {
            "baseline_mode": mode,
            "baseline_rgb": None,
            "baseline_blur_sigma": float(sigma),
        }

    baseline_tensor = (
        torch.from_numpy(np.asarray(baseline_np, dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=x.device, dtype=x.dtype)
    )
    return baseline_tensor, info


def baseline_fields_from_mapping(mapping: Mapping[str, object]):
    normalized = canonicalize_baseline_config(mapping)
    return {
        "baseline_mode": normalized.get("baseline_mode", DEFAULT_BASELINE_MODE),
        "baseline_rgb": normalized.get("baseline_rgb"),
        "baseline_blur_sigma": normalized.get("baseline_blur_sigma"),
    }
