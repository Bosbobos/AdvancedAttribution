from __future__ import annotations

"""Self-contained cheap-IG utilities for sparse neuron selection."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .config import AttackConfig


@dataclass(slots=True)
class SparseSelection:
    """Sparse neuron selection and detached weights for one attack iteration."""

    positive_mask: torch.Tensor
    negative_mask: torch.Tensor
    detached_ia: torch.Tensor
    detached_attr: torch.Tensor
    tail_scalar: torch.Tensor
    selected_positive_counts: list[int]
    selected_negative_counts: list[int]
    fill_betas: list[float]


def _alpha_grid(steps: int) -> np.ndarray:
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    return np.linspace(1.0 / steps, 1.0, steps, dtype=np.float64)


def _segment_index_mask(alphas: np.ndarray, start: float, end: float) -> np.ndarray:
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError(f"alpha segment must satisfy 0 <= start <= end <= 1, got [{start}, {end}]")
    if end < 1.0:
        return np.nonzero((alphas >= start) & (alphas < end))[0]
    return np.nonzero((alphas >= start) & (alphas <= end))[0]


def _selection_masks(
    flat_scores: np.ndarray,
    *,
    selection_mode: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for positive and negative selections per batch row."""

    batch_size, width = flat_scores.shape
    positive_mask = np.zeros((batch_size, width), dtype=bool)
    negative_mask = np.zeros((batch_size, width), dtype=bool)

    for row in range(batch_size):
        scores = flat_scores[row]
        if selection_mode in {"signed", "positive"}:
            pos_idx = np.flatnonzero(scores > 0.0)
            if pos_idx.size > 0:
                k = min(top_k, pos_idx.size)
                if k >= pos_idx.size:
                    positive_mask[row, pos_idx] = True
                else:
                    chosen = np.argpartition(scores[pos_idx], pos_idx.size - k)[-k:]
                    positive_mask[row, pos_idx[chosen]] = True
        if selection_mode == "signed":
            neg_idx = np.flatnonzero(scores < 0.0)
            if neg_idx.size > 0:
                k = min(top_k, neg_idx.size)
                if k >= neg_idx.size:
                    negative_mask[row, neg_idx] = True
                else:
                    chosen = np.argpartition(np.abs(scores[neg_idx]), neg_idx.size - k)[-k:]
                    negative_mask[row, neg_idx[chosen]] = True
        elif selection_mode == "unsigned":
            abs_idx = np.argsort(np.abs(scores))[-min(top_k, width) :]
            chosen_signs = scores[abs_idx] >= 0.0
            positive_mask[row, abs_idx[chosen_signs]] = True
            negative_mask[row, abs_idx[~chosen_signs]] = True

    return positive_mask, negative_mask


def _hybrid_fill_scalar(
    approx_vector: np.ndarray,
    selected_mask: np.ndarray,
    *,
    selection_mode: str,
    fill_mode: str,
    fill_rho: float,
    eps: float = 1e-12,
) -> tuple[float, float]:
    """Return a detached scalar tail for `naa_scaled` fill using cheap selected neurons only."""

    if fill_mode == "zero":
        return 0.0, 0.0

    approx_selected = approx_vector[selected_mask]
    if approx_selected.size == 0:
        return 0.0, 0.0

    if selection_mode == "positive":
        approx_scale = np.maximum(approx_selected, 0.0)
        tail_values = np.maximum(approx_vector[~selected_mask], 0.0)
    else:
        approx_scale = np.abs(approx_selected)
        tail_values = approx_vector[~selected_mask]

    valid = (approx_scale > eps) & np.isfinite(approx_scale)
    if not np.any(valid):
        return 0.0, 0.0

    residual_scale = np.abs(tail_values)
    residual_scale = residual_scale[np.isfinite(residual_scale) & (residual_scale > eps)]
    if residual_scale.size == 0:
        return 0.0, 0.0

    # Tail fill stays cheap-only: the selected NAA/cheap-IG-important neurons define
    # the admissible scale for the non-selected tail, without any exact segment/JVP.
    cap_reference = float(np.quantile(approx_scale[valid], 0.1))
    beta_cap = float(fill_rho) * cap_reference / float(residual_scale.max() + eps) if cap_reference > eps else 0.0
    beta = max(0.0, min(1.0, beta_cap))
    tail_scalar = float(beta * tail_values.sum())
    return tail_scalar, beta


def estimate_sparse_selection(
    *,
    model: torch.nn.Module,
    hook,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    target_classes: torch.Tensor,
    config: AttackConfig,
    split_output: Callable[[object], tuple[torch.Tensor | None, torch.Tensor]],
    unwrap_activation: Callable[[object], torch.Tensor],
) -> SparseSelection:
    """Estimate current sparse cheap-IG weights without building an outer optimization graph."""

    alphas = _alpha_grid(config.ia_steps)

    def forward_with_activation(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hook.clear()
        outputs = model(x)
        activation = unwrap_activation(hook.get())
        _, logits = split_output(outputs)
        return logits, activation

    delta_x = (inputs - baselines).contiguous()

    with torch.no_grad():
        _, act_x = forward_with_activation(inputs)
        _, act_0 = forward_with_activation(baselines)
        delta_y = (act_x - act_0).detach()

    sum_a_full = torch.zeros_like(delta_y)
    batch_indices = torch.arange(inputs.shape[0], device=inputs.device)

    for step_idx, alpha in enumerate(alphas):
        x_alpha = (baselines + float(alpha) * delta_x).contiguous().detach().requires_grad_(True)
        logits, activation = forward_with_activation(x_alpha)
        score = logits[batch_indices, target_classes].sum()
        grad_y = torch.autograd.grad(score, activation, retain_graph=False, create_graph=False)[0]
        sum_a_full += grad_y.detach()

        del x_alpha, logits, activation, score, grad_y

    approx_ia = sum_a_full / float(len(alphas))
    approx_attr = delta_y * approx_ia

    flat_ia = approx_ia.reshape(inputs.shape[0], -1).detach().cpu().numpy().astype(np.float64, copy=False)
    flat_attr = approx_attr.reshape(inputs.shape[0], -1).detach().cpu().numpy().astype(np.float64, copy=False)

    pos_mask_np, neg_mask_np = _selection_masks(
        flat_attr,
        selection_mode=config.selection_mode,
        top_k=config.selection_top_k,
    )
    selected_mask_np = pos_mask_np | neg_mask_np

    tail_scalars: list[float] = []
    fill_betas: list[float] = []
    for row in range(inputs.shape[0]):
        tail_scalar, beta = _hybrid_fill_scalar(
            flat_attr[row],
            selected_mask_np[row],
            selection_mode=config.selection_mode,
            fill_mode=config.fill_mode,
            fill_rho=config.fill_rho,
        )
        tail_scalars.append(tail_scalar)
        fill_betas.append(beta)

    device = inputs.device
    return SparseSelection(
        positive_mask=torch.from_numpy(pos_mask_np).to(device=device, dtype=torch.bool),
        negative_mask=torch.from_numpy(neg_mask_np).to(device=device, dtype=torch.bool),
        detached_ia=torch.from_numpy(flat_ia).to(device=device, dtype=inputs.dtype),
        detached_attr=torch.from_numpy(flat_attr).to(device=device, dtype=inputs.dtype),
        tail_scalar=torch.tensor(tail_scalars, device=device, dtype=inputs.dtype),
        selected_positive_counts=[int(mask.sum()) for mask in pos_mask_np],
        selected_negative_counts=[int(mask.sum()) for mask in neg_mask_np],
        fill_betas=fill_betas,
    )


def sparse_weighted_surrogate_loss(
    *,
    current_activation: torch.Tensor,
    baseline_activation: torch.Tensor,
    selection: SparseSelection,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a stable first-order surrogate from the current sparse cheap-IG estimate."""

    flat_delta_y = (current_activation - baseline_activation).reshape(current_activation.shape[0], -1)
    flat_attr = flat_delta_y * selection.detached_ia

    positive_term = (flat_attr * selection.positive_mask.to(dtype=flat_attr.dtype)).sum(dim=1)
    negative_term = (flat_attr * selection.negative_mask.to(dtype=flat_attr.dtype)).sum(dim=1)
    sample_loss = positive_term + gamma * negative_term + selection.tail_scalar
    return sample_loss.mean(), flat_attr.detach()
