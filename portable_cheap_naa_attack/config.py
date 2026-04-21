from __future__ import annotations

"""Typed configuration for the portable cheap-IG based NAA attack."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class AttackConfig:
    """Decision-complete configuration for a cheap-IG based feature attack."""

    weights_path: Path = _PACKAGE_ROOT / "weights" / "yolo11s-cls.pt"
    output_dir: Path = _PACKAGE_ROOT / "outputs"

    layer_name: str = "model.6"
    baseline_mode: str = "zero"
    image_size: int = 224

    segment_start: float = 0.0
    segment_end: float = 0.1
    selection_mode: str = "signed"
    selection_top_k: int = 8000
    fill_mode: str = "zero"
    fill_rho: float = 0.8
    ia_steps: int = 30

    epsilon: float = 16.0 / 255.0
    attack_steps: int = 10
    step_size: float = 1.6 / 255.0
    momentum: float = 1.0
    gamma: float = 1.0
    f_pos: str = "identity"
    f_neg: str = "identity"

    variant: str = "base"
    dim_prob: float = 0.7
    dim_resize_low: float = 1.0
    dim_resize_high: float = 1.1
    pim_amplification: float = 2.5
    pim_kernel_size: int = 3

    batch_size: int = 0
    top_n_preview: int = 5
    force_device: str | None = None
    fallback_to_cpu_for_attribution: bool = True
    live_plots: bool = True
    save_latest: bool = True
    save_best: bool = True
    save_history_json: bool = True
    verbose: bool = True

    def __post_init__(self) -> None:
        self.weights_path = Path(self.weights_path)
        self.output_dir = Path(self.output_dir)

        if self.layer_name.strip() == "":
            raise ValueError("layer_name must not be empty")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.ia_steps <= 0:
            raise ValueError(f"ia_steps must be positive, got {self.ia_steps}")
        if self.attack_steps <= 0:
            raise ValueError(f"attack_steps must be positive, got {self.attack_steps}")
        if self.selection_top_k <= 0:
            raise ValueError(f"selection_top_k must be positive, got {self.selection_top_k}")
        if not 0.0 <= self.segment_start <= self.segment_end <= 1.0:
            raise ValueError(
                "segment_start and segment_end must satisfy 0 <= start <= end <= 1, "
                f"got [{self.segment_start}, {self.segment_end}]"
            )
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.step_size <= 0.0:
            raise ValueError(f"step_size must be > 0, got {self.step_size}")
        if self.momentum < 0.0:
            raise ValueError(f"momentum must be >= 0, got {self.momentum}")
        if self.gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if self.dim_prob < 0.0 or self.dim_prob > 1.0:
            raise ValueError(f"dim_prob must be in [0, 1], got {self.dim_prob}")
        if self.dim_resize_low < 1.0:
            raise ValueError(f"dim_resize_low must be >= 1.0, got {self.dim_resize_low}")
        if self.dim_resize_high < self.dim_resize_low:
            raise ValueError(
                "dim_resize_high must be >= dim_resize_low, "
                f"got {self.dim_resize_high} < {self.dim_resize_low}"
            )
        if self.pim_amplification <= 0.0:
            raise ValueError(f"pim_amplification must be > 0, got {self.pim_amplification}")
        if self.pim_kernel_size <= 0 or self.pim_kernel_size % 2 == 0:
            raise ValueError(
                f"pim_kernel_size must be a positive odd integer, got {self.pim_kernel_size}"
            )
        if self.fill_mode not in {"zero", "naa_scaled"}:
            raise ValueError(f"Unsupported fill_mode: {self.fill_mode}")
        if self.selection_mode not in {"signed", "positive", "unsigned"}:
            raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")
        if self.variant not in {"base", "pd"}:
            raise ValueError(f"Unsupported variant: {self.variant}")
        if self.f_pos != "identity" or self.f_neg != "identity":
            raise ValueError("Only identity weighting is supported in this implementation")
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >= 0, got {self.batch_size}")
        if self.force_device is not None and self.force_device not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"force_device must be one of cpu/cuda/mps, got {self.force_device!r}")

        alpha_samples = [float(index + 1) / float(self.ia_steps) for index in range(self.ia_steps)]
        if self.segment_end < 1.0:
            segment_steps = sum(self.segment_start <= alpha < self.segment_end for alpha in alpha_samples)
        else:
            segment_steps = sum(self.segment_start <= alpha <= self.segment_end for alpha in alpha_samples)
        if segment_steps <= 0:
            raise ValueError(
                "The current ia_steps and alpha segment produce no segment samples. "
                f"Got ia_steps={self.ia_steps}, segment=[{self.segment_start}, {self.segment_end}]"
            )

    @property
    def latest_dir(self) -> Path:
        return self.output_dir / "latest"

    @property
    def best_dir(self) -> Path:
        return self.output_dir / "best"

    @property
    def effective_batch_size(self) -> int | None:
        return None if self.batch_size == 0 else self.batch_size

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["weights_path"] = str(self.weights_path)
        payload["output_dir"] = str(self.output_dir)
        payload["latest_dir"] = str(self.latest_dir)
        payload["best_dir"] = str(self.best_dir)
        return payload
