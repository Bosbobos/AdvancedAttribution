from __future__ import annotations

"""Reference NAA attack adapted to the self-contained portable package."""

import time
from typing import Any

import numpy as np
import torch

from ..attack_core import (
    AttackRunResult,
    LayerHook,
    _build_baselines,
    _clear_backend_cache,
    _forward_with_activation,
    _gather_target_probabilities,
    _importance_maps_from_flat_attr,
    _load_batch,
    _load_model,
    _project_linf,
    _resolve_device,
    _write_attack_images,
)
from ..config import AttackConfig
from ..transforms import apply_dim, apply_pim_to_gradient
from ..visualization import LiveAttackVisualizer, write_json


def _estimate_naa_aggregate_gradient(
    *,
    model: torch.nn.Module,
    hook: LayerHook,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    target_classes: torch.Tensor,
    num_ens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate the baseline feature map and the paper-style aggregated neuron weights."""

    with torch.no_grad():
        _, baseline_feature = _forward_with_activation(model, hook, baselines)
        baseline_feature = baseline_feature.detach()

    batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
    aggregate_gradient = torch.zeros_like(baseline_feature)
    for index in range(num_ens):
        alpha = float(index) / float(num_ens)
        scaled = baselines + alpha * (inputs - baselines)
        logits, activation = _forward_with_activation(model, hook, scaled.detach().requires_grad_(True))
        probs = torch.softmax(logits, dim=1)
        score = probs[batch_indices, target_classes].sum()
        grad_y = torch.autograd.grad(score, activation, retain_graph=False, create_graph=False)[0]
        aggregate_gradient += grad_y.detach()
        del logits, activation, probs, score, grad_y

    aggregate_gradient /= float(num_ens)
    return aggregate_gradient, baseline_feature


def run_naa_attack(image_paths: list[str], config: AttackConfig) -> AttackRunResult:
    """Run the reference NAA baseline on one batch of images."""

    run_start = time.perf_counter()
    if not image_paths:
        raise ValueError("image_paths must contain at least one image")

    effective_batch_size = config.effective_batch_size or len(image_paths)
    if effective_batch_size < len(image_paths):
        raise ValueError(
            "run_naa_attack expects one explicit batch per call. "
            "Use a batch of images directly instead of micro-batch merging for comparison plots."
        )

    device = _resolve_device(config.force_device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.latest_dir.mkdir(parents=True, exist_ok=True)
    config.best_dir.mkdir(parents=True, exist_ok=True)

    model, class_names = _load_model(config.weights_path, device)
    hook = LayerHook(model, config.layer_name)
    try:
        clean_batch, image_arrays, sample_ids = _load_batch(
            image_paths,
            image_size=config.image_size,
            device=device,
        )
        baselines = _build_baselines(clean_batch, image_arrays, config.baseline_mode)

        with torch.no_grad():
            clean_logits, clean_activation = _forward_with_activation(model, hook, clean_batch)
        target_classes = clean_logits.argmax(dim=1)
        clean_target_logits = clean_logits.gather(1, target_classes[:, None]).squeeze(1).detach()
        clean_target_probabilities = _gather_target_probabilities(clean_logits, target_classes).detach()

        attribution_start = time.perf_counter()
        agg_grad, baseline_feature = _estimate_naa_aggregate_gradient(
            model=model,
            hook=hook,
            inputs=clean_batch,
            baselines=baselines,
            target_classes=target_classes,
            num_ens=config.ia_steps,
        )
        attribution_runtime_seconds = time.perf_counter() - attribution_start
        clean_attr = ((clean_activation - baseline_feature) * agg_grad).reshape(clean_batch.shape[0], -1).detach()
        clean_importance_maps = _importance_maps_from_flat_attr(
            clean_attr,
            clean_activation,
            out_hw=tuple(clean_batch.shape[-2:]),
        )

        x_adv = clean_batch.detach().clone()
        best_images = x_adv.detach().clone()
        best_drops = torch.zeros(x_adv.shape[0], device=device, dtype=x_adv.dtype)
        best_steps = torch.zeros(x_adv.shape[0], device=device, dtype=torch.long)
        momentum_buffer = torch.zeros_like(x_adv)
        batch_indices = torch.arange(x_adv.shape[0], device=device)
        history: list[dict[str, Any]] = []
        latest_paths: list[str] = []
        best_paths: list[str] = []
        current_importance_maps: list[np.ndarray] | None = None
        visualizer = LiveAttackVisualizer(
            clean_images=clean_batch.detach(),
            sample_ids=sample_ids,
            enabled=config.live_plots,
            method_label="NAA",
        )

        for step in range(1, config.attack_steps + 1):
            x_adv_var = x_adv.detach().clone().requires_grad_(True)
            transformed_inputs = apply_dim(
                x_adv_var,
                probability=config.dim_prob,
                resize_low=config.dim_resize_low,
                resize_high=config.dim_resize_high,
            ) if config.variant == "pd" else x_adv_var

            _, activation_adv = _forward_with_activation(model, hook, transformed_inputs)
            attack_loss = ((activation_adv - baseline_feature) * agg_grad).sum()
            attack_loss.backward()
            if x_adv_var.grad is None:
                raise RuntimeError("The adversarial variable has no gradient")

            grad = x_adv_var.grad.detach()
            if config.variant == "pd":
                grad = apply_pim_to_gradient(
                    grad,
                    amplification_factor=config.pim_amplification,
                    kernel_size=config.pim_kernel_size,
                )

            grad = grad / grad.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
            momentum_buffer = config.momentum * momentum_buffer + grad
            x_adv = x_adv - config.step_size * momentum_buffer.sign()
            x_adv = _project_linf(x_adv, center=clean_batch, epsilon=config.epsilon).detach()

            with torch.no_grad():
                logits_eval, activation_eval = _forward_with_activation(model, hook, x_adv)
                current_target_logits = logits_eval[batch_indices, target_classes]
                current_target_probabilities = _gather_target_probabilities(logits_eval, target_classes)
                current_drops = clean_target_logits - current_target_logits
                improved = current_drops > best_drops
                best_drops = torch.where(improved, current_drops, best_drops)
                best_steps = torch.where(improved, torch.full_like(best_steps, step), best_steps)
                best_images = torch.where(improved[:, None, None, None], x_adv.detach(), best_images)
                latest_paths, best_paths = _write_attack_images(
                    config=config,
                    sample_ids=sample_ids,
                    current_images=x_adv.detach(),
                    best_images=best_images.detach(),
                )
                current_attr = ((activation_eval - baseline_feature) * agg_grad).reshape(x_adv.shape[0], -1).detach()
                current_importance_maps = _importance_maps_from_flat_attr(
                    current_attr,
                    activation_eval,
                    out_hw=tuple(clean_batch.shape[-2:]),
                )

            drop_list = current_drops.detach().cpu().tolist()
            confidence_list = current_target_probabilities.detach().cpu().tolist()
            best_index = int(torch.argmax(current_drops).item())
            worst_index = int(torch.argmin(current_drops).item())
            step_record = {
                "step": step,
                "attack_loss": float(attack_loss.detach().item()),
                "mean_target_logit_drop": float(np.mean(drop_list)),
                "best_confidence": float(confidence_list[best_index]),
                "worst_confidence": float(confidence_list[worst_index]),
                "elapsed_seconds": float(time.perf_counter() - run_start),
                "attribution_seconds": float(attribution_runtime_seconds),
                "target_logit_drop_per_sample": drop_list,
                "target_confidence_per_sample": confidence_list,
                "agg_grad_mean_abs": float(agg_grad.abs().mean().item()),
            }
            history.append(step_record)

            visualizer.update(
                history=history,
                current_images=x_adv.detach().cpu(),
                best_images=best_images.detach().cpu(),
                current_drops=drop_list,
                current_confidences=confidence_list,
                best_index=best_index,
                worst_index=worst_index,
                clean_importance_maps=clean_importance_maps,
                current_importance_maps=current_importance_maps,
            )

            if config.verbose:
                print(
                    f"[NAA step {step:02d}/{config.attack_steps}] "
                    f"loss={step_record['attack_loss']:.5f} "
                    f"mean_drop={step_record['mean_target_logit_drop']:.5f}"
                )

            _clear_backend_cache()

        with torch.no_grad():
            final_logits, _ = _forward_with_activation(model, hook, x_adv)
            best_logits, _ = _forward_with_activation(model, hook, best_images)

        final_target_logits = final_logits[batch_indices, target_classes]
        final_target_probabilities = _gather_target_probabilities(final_logits, target_classes)
        best_target_logits = best_logits[batch_indices, target_classes]
        best_target_probabilities = _gather_target_probabilities(best_logits, target_classes)

        result = AttackRunResult(
            image_paths=[str(path) for path in image_paths],
            sample_ids=sample_ids,
            device=str(device),
            target_classes=[int(value) for value in target_classes.detach().cpu().tolist()],
            target_names=[class_names[int(value)] for value in target_classes.detach().cpu().tolist()],
            clean_target_logits=[float(value) for value in clean_target_logits.detach().cpu().tolist()],
            clean_target_probabilities=[float(value) for value in clean_target_probabilities.detach().cpu().tolist()],
            final_target_logits=[float(value) for value in final_target_logits.detach().cpu().tolist()],
            final_target_probabilities=[
                float(value) for value in final_target_probabilities.detach().cpu().tolist()
            ],
            best_target_logits=[float(value) for value in best_target_logits.detach().cpu().tolist()],
            best_target_probabilities=[float(value) for value in best_target_probabilities.detach().cpu().tolist()],
            final_target_logit_drops=[
                float(value)
                for value in (clean_target_logits - final_target_logits).detach().cpu().tolist()
            ],
            best_target_logit_drops=[
                float(value)
                for value in (clean_target_logits - best_target_logits).detach().cpu().tolist()
            ],
            best_steps=[int(value) for value in best_steps.detach().cpu().tolist()],
            latest_image_paths=latest_paths,
            best_image_paths=best_paths,
            history=history,
            aggregate_history={
                "mean_target_logit_drop": [float(row["mean_target_logit_drop"]) for row in history],
                "best_confidence": [float(row["best_confidence"]) for row in history],
                "worst_confidence": [float(row["worst_confidence"]) for row in history],
            },
            config=config.to_dict(),
            method_name="NAA",
            total_runtime_seconds=time.perf_counter() - run_start,
            attribution_runtime_seconds=attribution_runtime_seconds,
            clean_importance_maps=clean_importance_maps,
            final_importance_maps=current_importance_maps,
        )
        if config.save_history_json:
            write_json(config.output_dir / "attack_history.json", result.to_dict())
        return result
    finally:
        hook.remove()
        _clear_backend_cache()
