from __future__ import annotations

"""Example presets for the portable cheap-IG NAA attack."""

from pathlib import Path

from .config import AttackConfig


def build_base_config() -> AttackConfig:
    return AttackConfig(
        variant="base",
        live_plots=True,
        save_history_json=True,
        selection_top_k=8000,
        fill_mode="zero",
    )


def build_pd_config() -> AttackConfig:
    return AttackConfig(
        variant="pd",
        live_plots=True,
        save_history_json=True,
        selection_top_k=8000,
        fill_mode="zero",
    )


def build_naa_base_config(*, output_dir: str | Path | None = None) -> AttackConfig:
    kwargs = {}
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    return AttackConfig(
        variant="base",
        live_plots=False,
        save_history_json=True,
        selection_top_k=8000,
        fill_mode="zero",
        **kwargs,
    )


def build_naa_pd_config(*, output_dir: str | Path | None = None) -> AttackConfig:
    kwargs = {}
    if output_dir is not None:
        kwargs["output_dir"] = output_dir
    return AttackConfig(
        variant="pd",
        live_plots=False,
        save_history_json=True,
        selection_top_k=8000,
        fill_mode="zero",
        **kwargs,
    )
