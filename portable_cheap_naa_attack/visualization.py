from __future__ import annotations

"""Live visualization helpers for notebook-first attack runs."""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import clear_output, display
from PIL import Image


def tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    """Convert a CHW image tensor in [0, 1] to an RGB array."""

    array = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.asarray(np.round(array * 255.0), dtype=np.uint8)


def save_image_tensor(image: torch.Tensor, path: Path) -> None:
    """Persist a tensor as a PNG image."""

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_rgb(image)).save(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Save a JSON payload with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _normalize_importance_map(map_2d: np.ndarray) -> np.ndarray:
    """Map signed importance values to a stable [-1, 1] range for plotting."""

    finite = map_2d[np.isfinite(map_2d)]
    if finite.size == 0:
        return np.zeros_like(map_2d, dtype=np.float32)

    scale = float(np.quantile(np.abs(finite), 0.995))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.max(np.abs(finite)))
    if not np.isfinite(scale) or scale <= 1e-12:
        return np.zeros_like(map_2d, dtype=np.float32)
    return np.clip(map_2d / scale, -1.0, 1.0).astype(np.float32, copy=False)


def project_importance_map(importance: torch.Tensor, out_hw: tuple[int, int]) -> np.ndarray:
    """Collapse a layer attribution tensor into a signed 2D map aligned to the image."""

    if importance.ndim == 4:
        if importance.shape[0] != 1:
            raise ValueError("project_importance_map expects a single sample when given a 4D tensor")
        importance = importance[0]
    if importance.ndim == 3:
        importance = importance.sum(dim=0)
    if importance.ndim != 2:
        raise ValueError(f"Expected a 2D/3D/4D importance tensor, got shape {tuple(importance.shape)}")

    importance_2d = importance.detach().to(dtype=torch.float32, device="cpu")[None, None, :, :]
    resized = F.interpolate(importance_2d, size=out_hw, mode="bilinear", align_corners=False)[0, 0]
    return _normalize_importance_map(resized.numpy())


class LiveAttackVisualizer:
    """Notebook-friendly live plots for both single-image and batch attack runs."""

    def __init__(
        self,
        *,
        clean_images: torch.Tensor,
        sample_ids: list[str],
        enabled: bool,
        method_label: str,
    ) -> None:
        self.enabled = enabled
        self.clean_images = clean_images.detach().cpu()
        self.sample_ids = sample_ids
        self.method_label = method_label
        self.fig = None
        self.axes = None

        if not self.enabled:
            return

        plt.ion()
        if clean_images.shape[0] == 1:
            self.fig, self.axes = plt.subplots(2, 4, figsize=(18, 9))
            self.fig.suptitle(f"{self.method_label} Attack Progress", fontsize=14)
        else:
            self.fig, self.axes = plt.subplots(2, 4, figsize=(18, 8))
            self.fig.suptitle(f"{self.method_label} Batch Progress", fontsize=14)

    def update(
        self,
        *,
        history: list[dict[str, Any]],
        current_images: torch.Tensor,
        best_images: torch.Tensor,
        current_drops: list[float],
        current_confidences: list[float],
        best_index: int,
        worst_index: int,
        clean_importance_maps: list[np.ndarray],
        current_importance_maps: list[np.ndarray],
    ) -> None:
        if not self.enabled or self.fig is None:
            return

        clear_output(wait=True)

        if current_images.shape[0] == 1:
            axes = self.axes
            assert axes is not None
            clean_ax = axes[0, 0]
            current_ax = axes[0, 1]
            best_ax = axes[0, 2]
            aux_ax = axes[0, 3]
            clean_map_ax = axes[1, 0]
            current_map_ax = axes[1, 1]
            drop_ax = axes[1, 2]
            conf_ax = axes[1, 3]

            clean_ax.clear()
            clean_ax.imshow(tensor_to_rgb(self.clean_images[0]))
            clean_ax.set_title(f"{self.method_label}: clean")
            clean_ax.axis("off")

            current_ax.clear()
            current_ax.imshow(tensor_to_rgb(current_images[0]))
            current_ax.set_title(f"{self.method_label}: current adversarial")
            current_ax.axis("off")

            best_ax.clear()
            best_ax.imshow(tensor_to_rgb(best_images[0]))
            best_ax.set_title(f"{self.method_label}: best by target-logit drop")
            best_ax.axis("off")

            aux_ax.clear()
            aux_ax.axis("off")

            clean_map_ax.clear()
            clean_map_ax.imshow(clean_importance_maps[0], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            clean_map_ax.set_title(f"{self.method_label}: clean target importance")
            clean_map_ax.axis("off")

            current_map_ax.clear()
            current_map_ax.imshow(current_importance_maps[0], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            current_map_ax.set_title(f"{self.method_label}: current target importance")
            current_map_ax.axis("off")

            steps = [row["step"] for row in history]
            drops = [row["mean_target_logit_drop"] for row in history]
            confidences = [row["best_confidence"] for row in history]

            drop_ax.clear()
            drop_ax.plot(steps, drops, marker="o")
            drop_ax.set_title(f"{self.method_label}: target logit drop")
            drop_ax.set_xlabel("Iteration")
            drop_ax.grid(alpha=0.3)

            conf_ax.clear()
            conf_ax.plot(steps, confidences, marker="o", color="tab:orange")
            conf_ax.set_title(f"{self.method_label}: target class confidence")
            conf_ax.set_xlabel("Iteration")
            conf_ax.grid(alpha=0.3)

            best_history_index = int(np.argmax(drops)) if drops else 0
            best_iteration = history[best_history_index]["step"] if drops else 0
            latest_elapsed = float(history[-1].get("elapsed_seconds", 0.0)) if history else 0.0
            latest_attribution = float(history[-1].get("attribution_seconds", 0.0)) if history else 0.0
            aux_ax.text(
                0.0,
                0.95,
                f"Method: {self.method_label}\n"
                f"Elapsed: {latest_elapsed:.2f}s\n"
                f"Attribution: {latest_attribution:.2f}s\n"
                "\n"
                f"Current drop: {current_drops[0]:.4f}\n"
                f"Current confidence: {current_confidences[0]:.4f}\n"
                f"Best iteration: {best_iteration}",
                va="top",
                ha="left",
                fontsize=11,
            )
        else:
            axes = self.axes
            assert axes is not None
            mean_ax = axes[0, 0]
            best_conf_ax = axes[0, 1]
            worst_conf_ax = axes[0, 2]
            info_ax = axes[0, 3]
            best_clean_map_ax = axes[1, 0]
            best_current_map_ax = axes[1, 1]
            worst_clean_map_ax = axes[1, 2]
            worst_current_map_ax = axes[1, 3]

            steps = [row["step"] for row in history]
            mean_drops = [row["mean_target_logit_drop"] for row in history]
            best_conf = [row["best_confidence"] for row in history]
            worst_conf = [row["worst_confidence"] for row in history]

            mean_ax.clear()
            mean_ax.plot(steps, mean_drops, marker="o")
            mean_ax.set_title(f"{self.method_label}: mean target-logit drop")
            mean_ax.set_xlabel("Iteration")
            mean_ax.grid(alpha=0.3)

            best_conf_ax.clear()
            best_conf_ax.plot(steps, best_conf, marker="o", color="tab:green")
            best_conf_ax.set_title(f"{self.method_label}: confidence of best-drop sample")
            best_conf_ax.set_xlabel("Iteration")
            best_conf_ax.grid(alpha=0.3)

            worst_conf_ax.clear()
            worst_conf_ax.plot(steps, worst_conf, marker="o", color="tab:red")
            worst_conf_ax.set_title(f"{self.method_label}: confidence of worst-drop sample")
            worst_conf_ax.set_xlabel("Iteration")
            worst_conf_ax.grid(alpha=0.3)

            info_ax.clear()
            info_ax.axis("off")
            latest_elapsed = float(history[-1].get("elapsed_seconds", 0.0)) if history else 0.0
            latest_attribution = float(history[-1].get("attribution_seconds", 0.0)) if history else 0.0
            info_ax.text(
                0.0,
                0.95,
                f"Method: {self.method_label}\n"
                f"Elapsed: {latest_elapsed:.2f}s\n"
                f"Attribution: {latest_attribution:.2f}s\n\n"
                f"Best-drop sample: {self.sample_ids[best_index]}\n"
                f"Current drop: {current_drops[best_index]:.4f}\n\n"
                f"Worst-drop sample: {self.sample_ids[worst_index]}\n"
                f"Current drop: {current_drops[worst_index]:.4f}",
                va="top",
                ha="left",
                fontsize=11,
            )

            best_clean_map_ax.clear()
            best_clean_map_ax.imshow(clean_importance_maps[best_index], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            best_clean_map_ax.set_title(f"{self.method_label}: best sample clean map")
            best_clean_map_ax.axis("off")

            best_current_map_ax.clear()
            best_current_map_ax.imshow(current_importance_maps[best_index], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            best_current_map_ax.set_title(f"{self.method_label}: best sample current map")
            best_current_map_ax.axis("off")

            worst_clean_map_ax.clear()
            worst_clean_map_ax.imshow(clean_importance_maps[worst_index], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            worst_clean_map_ax.set_title(f"{self.method_label}: worst sample clean map")
            worst_clean_map_ax.axis("off")

            worst_current_map_ax.clear()
            worst_current_map_ax.imshow(current_importance_maps[worst_index], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
            worst_current_map_ax.set_title(f"{self.method_label}: worst sample current map")
            worst_current_map_ax.axis("off")

        self.fig.tight_layout()
        display(self.fig)
        self.fig.canvas.draw_idle()
