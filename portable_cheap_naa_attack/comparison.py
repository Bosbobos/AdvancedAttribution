from __future__ import annotations

"""Comparison plots for cheap-IG and NAA attack runs."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _runtime_summary(result: Any) -> str:
    """Format a compact runtime summary for figure titles."""

    total = float(getattr(result, "total_runtime_seconds", 0.0))
    attribution = float(getattr(result, "attribution_runtime_seconds", 0.0))
    return f"total={total:.2f}s, attribution={attribution:.2f}s"


def plot_attack_comparison(
    cheap_result: Any,
    naa_result: Any,
    *,
    cheap_label: str = "Cheap-IG",
    naa_label: str = "NAA",
    cheap_color: str = "tab:blue",
    naa_color: str = "tab:orange",
) -> plt.Figure:
    """Overlay the history curves of the two methods with distinct colors."""

    if bool(cheap_result.history) != bool(naa_result.history):
        raise ValueError("Both results must either have per-step histories or neither of them must")
    if not cheap_result.history or not naa_result.history:
        raise ValueError("Comparison plotting requires per-step histories; micro-batch merged results are not supported")

    cheap_steps = [int(row["step"]) for row in cheap_result.history]
    naa_steps = [int(row["step"]) for row in naa_result.history]

    if len(cheap_result.sample_ids) == 1 and len(naa_result.sample_ids) == 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        cheap_drops = [float(row["mean_target_logit_drop"]) for row in cheap_result.history]
        naa_drops = [float(row["mean_target_logit_drop"]) for row in naa_result.history]
        axes[0].plot(cheap_steps, cheap_drops, marker="o", color=cheap_color, label=cheap_label)
        axes[0].plot(naa_steps, naa_drops, marker="o", color=naa_color, label=naa_label)
        axes[0].set_title("Target logit drop")
        axes[0].set_xlabel("Iteration")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        cheap_conf = [float(row["best_confidence"]) for row in cheap_result.history]
        naa_conf = [float(row["best_confidence"]) for row in naa_result.history]
        axes[1].plot(cheap_steps, cheap_conf, marker="o", color=cheap_color, label=cheap_label)
        axes[1].plot(naa_steps, naa_conf, marker="o", color=naa_color, label=naa_label)
        axes[1].set_title("Target class confidence")
        axes[1].set_xlabel("Iteration")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.suptitle(
            f"Cheap-IG vs NAA\n{cheap_label}: {_runtime_summary(cheap_result)} | "
            f"{naa_label}: {_runtime_summary(naa_result)}",
            fontsize=14,
        )
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    cheap_mean_drop = [float(row["mean_target_logit_drop"]) for row in cheap_result.history]
    naa_mean_drop = [float(row["mean_target_logit_drop"]) for row in naa_result.history]
    axes[0].plot(cheap_steps, cheap_mean_drop, marker="o", color=cheap_color, label=cheap_label)
    axes[0].plot(naa_steps, naa_mean_drop, marker="o", color=naa_color, label=naa_label)
    axes[0].set_title("Mean target-logit drop")
    axes[0].set_xlabel("Iteration")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    cheap_best_conf = [float(row["best_confidence"]) for row in cheap_result.history]
    naa_best_conf = [float(row["best_confidence"]) for row in naa_result.history]
    axes[1].plot(cheap_steps, cheap_best_conf, marker="o", color=cheap_color, label=cheap_label)
    axes[1].plot(naa_steps, naa_best_conf, marker="o", color=naa_color, label=naa_label)
    axes[1].set_title("Confidence of best-drop sample")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    cheap_worst_conf = [float(row["worst_confidence"]) for row in cheap_result.history]
    naa_worst_conf = [float(row["worst_confidence"]) for row in naa_result.history]
    axes[2].plot(cheap_steps, cheap_worst_conf, marker="o", color=cheap_color, label=cheap_label)
    axes[2].plot(naa_steps, naa_worst_conf, marker="o", color=naa_color, label=naa_label)
    axes[2].set_title("Confidence of worst-drop sample")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    fig.suptitle(
        f"Cheap-IG vs NAA\n{cheap_label}: {_runtime_summary(cheap_result)} | "
        f"{naa_label}: {_runtime_summary(naa_result)}",
        fontsize=14,
    )
    fig.tight_layout()
    return fig


def plot_importance_map_comparison(
    cheap_result: Any,
    naa_result: Any,
    *,
    cheap_label: str = "Cheap-IG",
    naa_label: str = "NAA",
) -> plt.Figure:
    """Render clean/final target importance maps for both methods."""

    if not cheap_result.clean_importance_maps or not cheap_result.final_importance_maps:
        raise ValueError("cheap_result does not contain importance maps")
    if not naa_result.clean_importance_maps or not naa_result.final_importance_maps:
        raise ValueError("naa_result does not contain importance maps")

    single_image = len(cheap_result.sample_ids) == 1 and len(naa_result.sample_ids) == 1
    if single_image:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        panels = [
            (axes[0, 0], cheap_result.clean_importance_maps[0], f"{cheap_label}: clean"),
            (axes[0, 1], cheap_result.final_importance_maps[0], f"{cheap_label}: final"),
            (axes[1, 0], naa_result.clean_importance_maps[0], f"{naa_label}: clean"),
            (axes[1, 1], naa_result.final_importance_maps[0], f"{naa_label}: final"),
        ]
    else:
        cheap_best = int(np.argmax(cheap_result.final_target_logit_drops))
        naa_best = int(np.argmax(naa_result.final_target_logit_drops))
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        panels = [
            (
                axes[0, 0],
                cheap_result.clean_importance_maps[cheap_best],
                f"{cheap_label}: clean ({cheap_result.sample_ids[cheap_best]})",
            ),
            (
                axes[0, 1],
                cheap_result.final_importance_maps[cheap_best],
                f"{cheap_label}: final ({cheap_result.sample_ids[cheap_best]})",
            ),
            (
                axes[1, 0],
                naa_result.clean_importance_maps[naa_best],
                f"{naa_label}: clean ({naa_result.sample_ids[naa_best]})",
            ),
            (
                axes[1, 1],
                naa_result.final_importance_maps[naa_best],
                f"{naa_label}: final ({naa_result.sample_ids[naa_best]})",
            ),
        ]

    for axis, importance_map, title in panels:
        axis.imshow(importance_map, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")

    fig.suptitle(
        f"Target-Class Importance Maps\n{cheap_label}: {_runtime_summary(cheap_result)} | "
        f"{naa_label}: {_runtime_summary(naa_result)}",
        fontsize=14,
    )
    fig.tight_layout()
    return fig
