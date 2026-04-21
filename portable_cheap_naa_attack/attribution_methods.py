from __future__ import annotations

"""Self-contained classifier attribution methods and visual benchmarking helpers."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.func import jvp

from .attack_core import (
    LayerHook,
    _build_baselines,
    _clear_backend_cache,
    _forward_with_activation,
    _gather_target_probabilities,
    _load_image,
    _load_model,
    _resolve_device,
)
from .visualization import project_importance_map


_PACKAGE_ROOT = Path(__file__).resolve().parent


@dataclass(slots=True)
class AttributionBenchmarkConfig:
    """Configuration for classifier-only attribution comparisons."""

    weights_path: Path = _PACKAGE_ROOT / "weights" / "yolo11s-cls.pt"
    layer_name: str = "model.6"
    baseline_mode: str = "zero"
    image_size: int = 224
    include_full_ig: bool = False
    n_steps: int = 128
    ranking_steps: int = 16
    segment_start: float = 0.0
    segment_end: float = 0.1
    selection_mode: str = "signed"
    selection_top_k: int = 8000
    fill_mode: str = "zero"
    fill_rho: float = 0.8
    clear_every: int = 8
    force_device: str | None = None

    def __post_init__(self) -> None:
        self.weights_path = Path(self.weights_path)
        if self.layer_name.strip() == "":
            raise ValueError("layer_name must not be empty")
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        if self.ranking_steps <= 0:
            raise ValueError(f"ranking_steps must be positive, got {self.ranking_steps}")
        if self.selection_top_k <= 0:
            raise ValueError(f"selection_top_k must be positive, got {self.selection_top_k}")
        if not 0.0 <= self.segment_start <= self.segment_end <= 1.0:
            raise ValueError(
                "segment_start and segment_end must satisfy 0 <= start <= end <= 1, "
                f"got [{self.segment_start}, {self.segment_end}]"
            )
        if self.selection_mode not in {"signed", "positive", "unsigned"}:
            raise ValueError(f"Unsupported selection_mode: {self.selection_mode}")
        if self.fill_mode not in {"zero", "naa_scaled"}:
            raise ValueError(f"Unsupported fill_mode: {self.fill_mode}")
        if self.clear_every < 0:
            raise ValueError(f"clear_every must be >= 0, got {self.clear_every}")

        segment_steps = len(_segment_alpha_values(self.n_steps, self.segment_start, self.segment_end))
        if segment_steps <= 0:
            raise ValueError(
                "The current n_steps and alpha segment produce no segment samples. "
                f"Got n_steps={self.n_steps}, segment=[{self.segment_start}, {self.segment_end}]"
            )
    @property
    def full_ig_label(self) -> str:
        return "Full IG [0,1]"

    @property
    def segment_ig_label(self) -> str:
        return f"Full IG [{self.segment_start:g},{self.segment_end:g}]"

    @property
    def enabled_method_names(self) -> list[str]:
        names: list[str] = []
        if self.include_full_ig:
            names.append(self.full_ig_label)
        names.extend(
            [
                self.segment_ig_label,
                "NAA",
                "Old Cheap-IG",
                "New Cheap-IG",
            ]
        )
        return names


@dataclass(slots=True)
class AttributionRuntime:
    """Reusable model state for repeated classifier attribution calls."""

    device: torch.device
    model: torch.nn.Module
    class_names: dict[int, str]
    hook: LayerHook

    def close(self) -> None:
        self.hook.remove()
        _clear_backend_cache()


@dataclass(slots=True)
class PreparedClassifierSample:
    """Prepared single-image inputs shared by all compared attribution methods."""

    image_path: str
    sample_id: str
    image_tensor: torch.Tensor
    baseline_tensor: torch.Tensor
    image_rgb: np.ndarray
    image_hw: tuple[int, int]
    delta_x: torch.Tensor
    clean_activation: torch.Tensor
    baseline_activation: torch.Tensor
    delta_y_flat: np.ndarray
    activation_shape: tuple[int, ...]
    target_class: int
    target_name: str
    target_logit: float
    target_probability: float
    baseline_target_logit: float
    fx_delta: float


@dataclass(slots=True)
class AttributionMethodResult:
    """One method applied to one image."""

    method_name: str
    target_class: int
    target_name: str
    target_logit: float
    target_probability: float
    fx_delta: float
    vector_sum: float
    abs_error: float
    runtime_seconds: float
    selected_neurons: int | None
    segment_steps: int
    fill_beta: float | None
    importance_map: np.ndarray
    cond_tensor: torch.Tensor


@dataclass(slots=True)
class AttributionComparisonResult:
    """All attribution methods evaluated on one image."""

    image_path: str
    sample_id: str
    image_rgb: np.ndarray
    target_class: int
    target_name: str
    method_results: dict[str, AttributionMethodResult]


def _alpha_grid(n_steps: int) -> np.ndarray:
    return np.linspace(1.0 / int(n_steps), 1.0, int(n_steps), dtype=np.float64)


def _segment_alpha_indices(n_steps: int, start: float, end: float) -> np.ndarray:
    alphas = _alpha_grid(n_steps)
    if end < 1.0:
        return np.nonzero((alphas >= start) & (alphas < end))[0]
    return np.nonzero((alphas >= start) & (alphas <= end))[0]


def _segment_alpha_values(n_steps: int, start: float, end: float) -> np.ndarray:
    alphas = _alpha_grid(n_steps)
    return alphas[_segment_alpha_indices(n_steps, start, end)]


def _selection_mask(scores: np.ndarray, selection_mode: str, selection_top_k: int) -> np.ndarray:
    """Replicate the legacy cheap-IG top-k selection used in earlier benchmarks."""

    scores = np.asarray(scores, dtype=np.float64)
    mask = np.zeros(scores.shape[0], dtype=bool)
    if selection_top_k <= 0 or scores.size == 0:
        return mask

    selection_top_k = int(min(selection_top_k, scores.size))

    if selection_mode == "signed":
        pos_idx = np.flatnonzero(scores > 0.0)
        neg_idx = np.flatnonzero(scores < 0.0)

        if pos_idx.size > 0:
            k_pos = min(selection_top_k, pos_idx.size)
            if k_pos >= pos_idx.size:
                mask[pos_idx] = True
            else:
                chosen_pos = np.argpartition(scores[pos_idx], pos_idx.size - k_pos)[-k_pos:]
                mask[pos_idx[chosen_pos]] = True

        if neg_idx.size > 0:
            k_neg = min(selection_top_k, neg_idx.size)
            if k_neg >= neg_idx.size:
                mask[neg_idx] = True
            else:
                chosen_neg = np.argpartition(np.abs(scores[neg_idx]), neg_idx.size - k_neg)[-k_neg:]
                mask[neg_idx[chosen_neg]] = True
        return mask

    if selection_mode == "unsigned":
        if selection_top_k >= scores.size:
            mask[:] = True
        else:
            chosen = np.argpartition(np.abs(scores), scores.size - selection_top_k)[-selection_top_k:]
            mask[chosen] = True
        return mask

    if selection_mode == "positive":
        pos_idx = np.flatnonzero(scores > 0.0)
        if pos_idx.size == 0:
            return mask
        k_pos = min(selection_top_k, pos_idx.size)
        if k_pos >= pos_idx.size:
            mask[pos_idx] = True
        else:
            chosen_pos = np.argpartition(scores[pos_idx], pos_idx.size - k_pos)[-k_pos:]
            mask[pos_idx[chosen_pos]] = True
        return mask

    raise ValueError(f"Unsupported selection_mode: {selection_mode}")


def _normalize_fill_mode(fill_mode: str) -> str:
    fill_mode = str(fill_mode).strip().lower()
    if fill_mode not in {"zero", "naa_scaled"}:
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")
    return fill_mode


def _scale_basis(values: np.ndarray, selection_mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if selection_mode == "positive":
        return np.maximum(arr, 0.0)
    return np.abs(arr)


def _residual_fill_values(values: np.ndarray, selection_mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if selection_mode == "positive":
        return np.maximum(arr, 0.0)
    return arr


def _hybrid_fill_vector(
    exact_vector: np.ndarray,
    approx_vector: np.ndarray,
    selected_mask: np.ndarray,
    *,
    selection_mode: str,
    fill_mode: str,
    fill_rho: float,
    cap_quantile: float = 0.1,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Legacy cheap-IG tail fill: exact selected neurons plus cheap approximation in the tail."""

    fill_mode = _normalize_fill_mode(fill_mode)
    exact = np.asarray(exact_vector, dtype=np.float64)
    approx = np.asarray(approx_vector, dtype=np.float64)
    mask = np.asarray(selected_mask, dtype=bool)

    filled = np.zeros_like(exact, dtype=np.float64)
    filled[mask] = exact[mask]
    stats = {"fill_beta": 0.0}

    if fill_mode == "zero" or exact.size == 0 or (~mask).sum() == 0:
        return filled, stats

    exact_scale = _scale_basis(exact[mask], selection_mode)
    approx_scale_selected = _scale_basis(approx[mask], selection_mode)
    valid = (
        np.isfinite(exact_scale)
        & np.isfinite(approx_scale_selected)
        & (exact_scale > eps)
        & (approx_scale_selected > eps)
    )
    if not np.any(valid):
        return filled, stats

    beta_raw = float(np.median(exact_scale[valid] / (approx_scale_selected[valid] + eps)))
    residual_values = _residual_fill_values(approx, selection_mode)
    residual_scale = _scale_basis(approx[~mask], selection_mode)
    residual_scale = residual_scale[np.isfinite(residual_scale) & (residual_scale > eps)]
    if residual_scale.size == 0:
        stats["fill_beta"] = max(0.0, beta_raw)
        return filled, stats

    q_in = float(np.quantile(exact_scale[valid], float(cap_quantile)))
    max_out = float(residual_scale.max())
    beta_cap = float(fill_rho) * q_in / (max_out + eps) if q_in > eps else 0.0
    beta = max(0.0, float(min(beta_raw, beta_cap)))

    filled[~mask] = beta * residual_values[~mask]
    stats["fill_beta"] = beta
    return filled, stats


def create_attribution_runtime(config: AttributionBenchmarkConfig) -> AttributionRuntime:
    """Load the classifier once and reuse it across all benchmarked attribution methods."""

    device = _resolve_device(config.force_device)
    model, class_names = _load_model(config.weights_path, device)
    hook = LayerHook(model, config.layer_name)
    return AttributionRuntime(device=device, model=model, class_names=class_names, hook=hook)


def prepare_classifier_sample(
    image_path: str | Path,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
) -> PreparedClassifierSample:
    """Load one image and fix the clean target class shared by all compared methods."""

    image_tensor_chw, image_rgb = _load_image(image_path, config.image_size)
    image_tensor = image_tensor_chw.unsqueeze(0).to(device=runtime.device, dtype=torch.float32)
    baselines = _build_baselines(image_tensor, [image_rgb], config.baseline_mode)

    with torch.no_grad():
        clean_logits, clean_activation = _forward_with_activation(runtime.model, runtime.hook, image_tensor)
        baseline_logits, baseline_activation = _forward_with_activation(runtime.model, runtime.hook, baselines)

    target_class_tensor = clean_logits.argmax(dim=1)
    target_class = int(target_class_tensor.item())
    target_name = runtime.class_names[target_class]
    target_logit = float(clean_logits[0, target_class].item())
    target_probability = float(_gather_target_probabilities(clean_logits, target_class_tensor)[0].item())
    baseline_target_logit = float(baseline_logits[0, target_class].item())
    fx_delta = float(target_logit - baseline_target_logit)

    return PreparedClassifierSample(
        image_path=str(image_path),
        sample_id=Path(image_path).stem,
        image_tensor=image_tensor,
        baseline_tensor=baselines,
        image_rgb=image_rgb,
        image_hw=tuple(image_rgb.shape[:2]),
        delta_x=(image_tensor - baselines).contiguous(),
        clean_activation=clean_activation.detach(),
        baseline_activation=baseline_activation.detach(),
        delta_y_flat=(clean_activation - baseline_activation).detach().reshape(-1).cpu().numpy().astype(np.float64),
        activation_shape=tuple(int(v) for v in clean_activation.shape[1:]),
        target_class=target_class,
        target_name=target_name,
        target_logit=target_logit,
        target_probability=target_probability,
        baseline_target_logit=baseline_target_logit,
        fx_delta=fx_delta,
    )


def _maybe_clear_cache(config: AttributionBenchmarkConfig, step_index: int) -> None:
    if config.clear_every > 0 and ((step_index + 1) % config.clear_every == 0):
        _clear_backend_cache()


def _build_method_result(
    *,
    method_name: str,
    sample: PreparedClassifierSample,
    attr_vector: np.ndarray,
    runtime_seconds: float,
    segment_steps: int,
    selected_neurons: int | None,
    fill_beta: float | None = None,
) -> AttributionMethodResult:
    """Convert a flat neuron attribution vector into the notebook-friendly result payload."""

    cond_tensor = torch.from_numpy(attr_vector.astype(np.float32, copy=False)).reshape(
        1, *sample.activation_shape
    )
    importance_map = project_importance_map(cond_tensor, sample.image_hw)
    vector_sum = float(attr_vector.sum())
    return AttributionMethodResult(
        method_name=method_name,
        target_class=sample.target_class,
        target_name=sample.target_name,
        target_logit=sample.target_logit,
        target_probability=sample.target_probability,
        fx_delta=sample.fx_delta,
        vector_sum=vector_sum,
        abs_error=float(abs(sample.fx_delta - vector_sum)),
        runtime_seconds=float(runtime_seconds),
        selected_neurons=selected_neurons,
        segment_steps=int(segment_steps),
        fill_beta=fill_beta,
        importance_map=importance_map,
        cond_tensor=cond_tensor,
    )


def _compute_approx_naa_vector(
    sample: PreparedClassifierSample,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
) -> np.ndarray:
    """Compute the dense NAA/cheap ranking vector over the full interpolation path."""

    alphas = _alpha_grid(config.ranking_steps)
    sum_a_full: np.ndarray | None = None

    for step_index, alpha in enumerate(alphas):
        x_alpha = (sample.baseline_tensor + float(alpha) * sample.delta_x).contiguous().detach().requires_grad_(True)
        logits, activation = _forward_with_activation(runtime.model, runtime.hook, x_alpha)
        score = logits[0, sample.target_class]
        grad_y = torch.autograd.grad(score, activation, retain_graph=False, create_graph=False)[0]
        grad_flat = grad_y.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)

        if sum_a_full is None:
            sum_a_full = np.zeros_like(grad_flat, dtype=np.float64)
        sum_a_full += grad_flat

        del x_alpha, logits, activation, score, grad_y, grad_flat
        runtime.hook.clear()
        _maybe_clear_cache(config, step_index)

    if sum_a_full is None:
        raise RuntimeError("NAA approximation pass produced no gradients")
    return (sum_a_full / float(len(alphas))) * sample.delta_y_flat


def _compute_exact_vector(
    sample: PreparedClassifierSample,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
    *,
    start: float,
    end: float,
) -> tuple[np.ndarray, int]:
    """Compute dense layer conductance on the requested alpha segment."""

    segment_alphas = _segment_alpha_values(config.n_steps, start, end)
    if segment_alphas.size == 0:
        raise ValueError(f"alpha segment [{start}, {end}] has no samples for n_steps={config.n_steps}")

    sum_ab: np.ndarray | None = None

    def layer_activation(x_in: torch.Tensor) -> torch.Tensor:
        _, activation = _forward_with_activation(runtime.model, runtime.hook, x_in)
        return activation

    for step_index, alpha in enumerate(segment_alphas):
        x_alpha = (sample.baseline_tensor + float(alpha) * sample.delta_x).contiguous().detach().requires_grad_(True)
        logits, activation = _forward_with_activation(runtime.model, runtime.hook, x_alpha)
        score = logits[0, sample.target_class]
        grad_y = torch.autograd.grad(score, activation, retain_graph=False, create_graph=False)[0]
        _, jvp_out = jvp(layer_activation, (x_alpha,), (sample.delta_x,))

        grad_flat = grad_y.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        dir_flat = jvp_out.detach().reshape(-1).cpu().numpy().astype(np.float64, copy=False)
        if sum_ab is None:
            sum_ab = np.zeros_like(grad_flat, dtype=np.float64)
        sum_ab += grad_flat * dir_flat

        del x_alpha, logits, activation, score, grad_y, jvp_out, grad_flat, dir_flat
        runtime.hook.clear()
        _maybe_clear_cache(config, step_index)

    if sum_ab is None:
        raise RuntimeError("Exact IG pass produced no contributions")
    return sum_ab / float(len(segment_alphas)), int(len(segment_alphas))


def _compute_exact_vector_for_selected_neurons(
    sample: PreparedClassifierSample,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
    *,
    selected_mask: np.ndarray,
    start: float,
    end: float,
) -> tuple[np.ndarray, int]:
    """Compute exact segment IG only for the selected neurons and keep zeros elsewhere."""

    selected_indices = np.flatnonzero(np.asarray(selected_mask, dtype=bool))
    segment_alphas = _segment_alpha_values(config.n_steps, start, end)
    if segment_alphas.size == 0:
        raise ValueError(f"alpha segment [{start}, {end}] has no samples for n_steps={config.n_steps}")
    if selected_indices.size == 0:
        return np.zeros_like(sample.delta_y_flat, dtype=np.float64), int(len(segment_alphas))

    selected_index_tensor = torch.from_numpy(selected_indices).to(device=runtime.device, dtype=torch.long)
    sum_selected = np.zeros(selected_indices.size, dtype=np.float64)

    def selected_layer_activation(x_in: torch.Tensor) -> torch.Tensor:
        _, activation = _forward_with_activation(runtime.model, runtime.hook, x_in)
        return activation.reshape(-1).index_select(0, selected_index_tensor)

    for step_index, alpha in enumerate(segment_alphas):
        x_alpha = (sample.baseline_tensor + float(alpha) * sample.delta_x).contiguous().detach().requires_grad_(True)
        logits, activation = _forward_with_activation(runtime.model, runtime.hook, x_alpha)
        score = logits[0, sample.target_class]
        grad_y = torch.autograd.grad(score, activation, retain_graph=False, create_graph=False)[0]
        grad_selected = grad_y.detach().reshape(-1).index_select(0, selected_index_tensor)

        runtime.hook.clear()
        _, jvp_out = jvp(selected_layer_activation, (x_alpha,), (sample.delta_x,))
        dir_selected = jvp_out.detach().reshape(-1)
        sum_selected += (grad_selected * dir_selected).cpu().numpy().astype(np.float64, copy=False)

        del x_alpha, logits, activation, score, grad_y, grad_selected, jvp_out, dir_selected
        runtime.hook.clear()
        _maybe_clear_cache(config, step_index)

    exact_vector = np.zeros_like(sample.delta_y_flat, dtype=np.float64)
    exact_vector[selected_indices] = sum_selected / float(len(segment_alphas))
    return exact_vector, int(len(segment_alphas))


def compute_full_ig(
    sample: PreparedClassifierSample,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
) -> AttributionMethodResult:
    """Dense IG conductance across the full interpolation path."""

    start = perf_counter()
    exact_vector, segment_steps = _compute_exact_vector(sample, runtime, config, start=0.0, end=1.0)
    return _build_method_result(
        method_name=config.full_ig_label,
        sample=sample,
        attr_vector=exact_vector,
        runtime_seconds=perf_counter() - start,
        segment_steps=segment_steps,
        selected_neurons=None,
    )


def compute_full_segment_ig(
    sample: PreparedClassifierSample,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
) -> AttributionMethodResult:
    """Dense IG conductance restricted to the configured alpha segment."""

    start = perf_counter()
    exact_vector, segment_steps = _compute_exact_vector(
        sample,
        runtime,
        config,
        start=config.segment_start,
        end=config.segment_end,
    )
    return _build_method_result(
        method_name=config.segment_ig_label,
        sample=sample,
        attr_vector=exact_vector,
        runtime_seconds=perf_counter() - start,
        segment_steps=segment_steps,
        selected_neurons=None,
    )


def compute_naa(
    sample: PreparedClassifierSample,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
) -> AttributionMethodResult:
    """Dense NAA approximation over the full path."""

    start = perf_counter()
    approx_vector = _compute_approx_naa_vector(sample, runtime, config)
    return _build_method_result(
        method_name="NAA",
        sample=sample,
        attr_vector=approx_vector,
        runtime_seconds=perf_counter() - start,
        segment_steps=config.ranking_steps,
        selected_neurons=None,
    )


def compute_old_cheap_ig(
    sample: PreparedClassifierSample,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
) -> AttributionMethodResult:
    """Legacy cheap-IG with cheap full-path ranking and dense exact segment pass."""

    start = perf_counter()
    approx_vector = _compute_approx_naa_vector(sample, runtime, config)
    exact_segment_vector, segment_steps = _compute_exact_vector(
        sample,
        runtime,
        config,
        start=config.segment_start,
        end=config.segment_end,
    )
    selected_mask = _selection_mask(approx_vector, config.selection_mode, config.selection_top_k)
    cheap_vector, fill_stats = _hybrid_fill_vector(
        exact_segment_vector,
        approx_vector,
        selected_mask,
        selection_mode=config.selection_mode,
        fill_mode=config.fill_mode,
        fill_rho=config.fill_rho,
    )
    return _build_method_result(
        method_name="Old Cheap-IG",
        sample=sample,
        attr_vector=cheap_vector,
        runtime_seconds=perf_counter() - start,
        segment_steps=segment_steps,
        selected_neurons=int(selected_mask.sum()),
        fill_beta=float(fill_stats["fill_beta"]),
    )


def compute_new_cheap_ig(
    sample: PreparedClassifierSample,
    runtime: AttributionRuntime,
    config: AttributionBenchmarkConfig,
) -> AttributionMethodResult:
    """Correct cheap-IG: rank densely with NAA, then run exact IG only for the selected neurons."""

    start = perf_counter()
    approx_vector = _compute_approx_naa_vector(sample, runtime, config)
    selected_mask = _selection_mask(approx_vector, config.selection_mode, config.selection_top_k)
    exact_selected_vector, segment_steps = _compute_exact_vector_for_selected_neurons(
        sample,
        runtime,
        config,
        selected_mask=selected_mask,
        start=config.segment_start,
        end=config.segment_end,
    )
    cheap_vector, fill_stats = _hybrid_fill_vector(
        exact_selected_vector,
        approx_vector,
        selected_mask,
        selection_mode=config.selection_mode,
        fill_mode=config.fill_mode,
        fill_rho=config.fill_rho,
    )
    return _build_method_result(
        method_name="New Cheap-IG",
        sample=sample,
        attr_vector=cheap_vector,
        runtime_seconds=perf_counter() - start,
        segment_steps=segment_steps,
        selected_neurons=int(selected_mask.sum()),
        fill_beta=float(fill_stats["fill_beta"]),
    )


def run_classifier_attribution_comparison(
    image_path: str | Path,
    config: AttributionBenchmarkConfig,
    *,
    runtime: AttributionRuntime | None = None,
) -> AttributionComparisonResult:
    """Run all classifier attribution methods on one image and collect visual/timing outputs."""

    owns_runtime = runtime is None
    runtime = runtime or create_attribution_runtime(config)
    try:
        sample = prepare_classifier_sample(image_path, runtime, config)
        method_results: dict[str, AttributionMethodResult] = {}
        if config.include_full_ig:
            method_results[config.full_ig_label] = compute_full_ig(sample, runtime, config)
        method_results[config.segment_ig_label] = compute_full_segment_ig(sample, runtime, config)
        method_results["NAA"] = compute_naa(sample, runtime, config)
        method_results["Old Cheap-IG"] = compute_old_cheap_ig(sample, runtime, config)
        method_results["New Cheap-IG"] = compute_new_cheap_ig(sample, runtime, config)
        return AttributionComparisonResult(
            image_path=sample.image_path,
            sample_id=sample.sample_id,
            image_rgb=sample.image_rgb,
            target_class=sample.target_class,
            target_name=sample.target_name,
            method_results=method_results,
        )
    finally:
        if owns_runtime:
            runtime.close()


def benchmark_classifier_attribution_runtimes(
    image_paths: list[str | Path],
    config: AttributionBenchmarkConfig,
    *,
    repeats: int = 1,
) -> dict[str, dict[str, float | list[float]]]:
    """Benchmark only classifier attribution runtime; the notebook can turn this into one final barplot."""

    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    runtime = create_attribution_runtime(config)
    method_names = config.enabled_method_names
    timings = {name: [] for name in method_names}

    try:
        for image_path in image_paths:
            sample = prepare_classifier_sample(image_path, runtime, config)
            runners: list[tuple[str, Callable[[], AttributionMethodResult]]] = []
            if config.include_full_ig:
                runners.append((config.full_ig_label, lambda sample=sample: compute_full_ig(sample, runtime, config)))
            runners.extend(
                [
                    (
                        config.segment_ig_label,
                        lambda sample=sample: compute_full_segment_ig(sample, runtime, config),
                    ),
                    ("NAA", lambda sample=sample: compute_naa(sample, runtime, config)),
                    ("Old Cheap-IG", lambda sample=sample: compute_old_cheap_ig(sample, runtime, config)),
                    ("New Cheap-IG", lambda sample=sample: compute_new_cheap_ig(sample, runtime, config)),
                ]
            )
            for _ in range(repeats):
                for method_name, runner in runners:
                    result = runner()
                    timings[method_name].append(float(result.runtime_seconds))
    finally:
        runtime.close()

    summary: dict[str, dict[str, float | list[float]]] = {}
    for method_name, values in timings.items():
        array = np.asarray(values, dtype=np.float64)
        summary[method_name] = {
            "times": values,
            "mean_seconds": float(array.mean()) if array.size else float("nan"),
            "std_seconds": float(array.std(ddof=0)) if array.size else float("nan"),
            "num_runs": int(array.size),
        }
    return summary


def plot_classifier_runtime_barplot(
    runtime_summary: dict[str, dict[str, float | list[float]]],
    *,
    title: str = "Classifier Attribution Runtime",
) -> plt.Figure:
    """Render the single final runtime barplot requested by the user."""

    method_names = list(runtime_summary)
    means = [float(runtime_summary[name]["mean_seconds"]) for name in method_names]
    stds = [float(runtime_summary[name]["std_seconds"]) for name in method_names]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["tab:blue", "tab:cyan", "tab:orange", "tab:red", "tab:green"]
    bars = ax.bar(method_names, means, yerr=stds, color=colors[: len(method_names)], capsize=5)
    ax.set_ylabel("Seconds")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", rotation=15)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, mean, f"{mean:.2f}s", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    return fig


def plot_visual_examples_table(
    comparisons: list[AttributionComparisonResult],
    config: AttributionBenchmarkConfig,
) -> plt.Figure:
    """Render a compact table of attribution overlays for several images and methods."""

    method_order = config.enabled_method_names
    rows = len(comparisons)
    cols = 1 + len(method_order)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.0 * rows), squeeze=False)

    for row_index, comparison in enumerate(comparisons):
        original_ax = axes[row_index, 0]
        original_ax.imshow(comparison.image_rgb)
        original_ax.set_title(f"Input\n{comparison.sample_id}\n{comparison.target_name}")
        original_ax.axis("off")

        for col_index, method_name in enumerate(method_order, start=1):
            result = comparison.method_results[method_name]
            axis = axes[row_index, col_index]
            axis.imshow(comparison.image_rgb)
            axis.imshow(result.importance_map, cmap="RdBu_r", vmin=-1.0, vmax=1.0, alpha=0.45)
            title = f"{method_name}\n{result.runtime_seconds:.2f}s"
            if result.selected_neurons is not None:
                title += f"\nsel={result.selected_neurons}"
            axis.set_title(title)
            axis.axis("off")

    fig.suptitle("Classifier Attribution Visual Examples", fontsize=14)
    fig.tight_layout()
    return fig
