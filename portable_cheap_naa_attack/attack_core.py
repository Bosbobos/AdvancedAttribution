from __future__ import annotations

"""Portable cheap-IG based NAA-style classifier attack."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gc
import time

import numpy as np
import torch
from PIL import Image, ImageFilter
from ultralytics import YOLO

from .cheap_ig_core import estimate_sparse_selection, sparse_weighted_surrogate_loss
from .config import AttackConfig
from .transforms import apply_dim, apply_pim_to_gradient
from .visualization import LiveAttackVisualizer, project_importance_map, save_image_tensor, write_json


@dataclass(slots=True)
class AttackRunResult:
    """Outputs of one attack run over a single batch of images."""

    image_paths: list[str]
    sample_ids: list[str]
    device: str
    target_classes: list[int]
    target_names: list[str]
    clean_target_logits: list[float]
    clean_target_probabilities: list[float]
    final_target_logits: list[float]
    final_target_probabilities: list[float]
    best_target_logits: list[float]
    best_target_probabilities: list[float]
    final_target_logit_drops: list[float]
    best_target_logit_drops: list[float]
    best_steps: list[int]
    latest_image_paths: list[str]
    best_image_paths: list[str]
    history: list[dict[str, Any]]
    aggregate_history: dict[str, list[float]]
    config: dict[str, Any]
    method_name: str
    total_runtime_seconds: float
    attribution_runtime_seconds: float
    chunk_results: list[dict[str, Any]] | None = None
    clean_importance_maps: list[np.ndarray] | None = None
    final_importance_maps: list[np.ndarray] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_paths": self.image_paths,
            "sample_ids": self.sample_ids,
            "device": self.device,
            "target_classes": self.target_classes,
            "target_names": self.target_names,
            "clean_target_logits": self.clean_target_logits,
            "clean_target_probabilities": self.clean_target_probabilities,
            "final_target_logits": self.final_target_logits,
            "final_target_probabilities": self.final_target_probabilities,
            "best_target_logits": self.best_target_logits,
            "best_target_probabilities": self.best_target_probabilities,
            "final_target_logit_drops": self.final_target_logit_drops,
            "best_target_logit_drops": self.best_target_logit_drops,
            "best_steps": self.best_steps,
            "latest_image_paths": self.latest_image_paths,
            "best_image_paths": self.best_image_paths,
            "history": self.history,
            "aggregate_history": self.aggregate_history,
            "config": self.config,
            "method_name": self.method_name,
            "total_runtime_seconds": self.total_runtime_seconds,
            "attribution_runtime_seconds": self.attribution_runtime_seconds,
            "chunk_results": self.chunk_results,
        }


class LayerHook:
    """Forward hook that captures the latest tensor activation for one layer."""

    def __init__(self, model: torch.nn.Module, layer_name: str) -> None:
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Layer {layer_name!r} was not found in model.named_modules()")
        self.layer_name = layer_name
        self.layer_store: dict[str, object] = {}
        self.handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output) -> None:
        self.layer_store[self.layer_name] = output

    def clear(self) -> None:
        self.layer_store.clear()

    def get(self) -> object:
        return self.layer_store[self.layer_name]

    def remove(self) -> None:
        self.handle.remove()


def _clear_backend_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _resolve_device(force_device: str | None) -> torch.device:
    if force_device is not None:
        if force_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("force_device='cuda' was requested but CUDA is not available")
        if force_device == "mps" and not torch.mps.is_available():
            raise RuntimeError("force_device='mps' was requested but MPS is not available")
        return torch.device(force_device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _split_classifier_output(output: object) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(output, (tuple, list)):
        if len(output) >= 2 and torch.is_tensor(output[1]):
            return output[0], output[1]
        if len(output) >= 1 and torch.is_tensor(output[0]):
            return None, output[0]
    if torch.is_tensor(output):
        return None, output
    raise TypeError(f"Unable to interpret classifier output of type {type(output).__name__}")


def _unwrap_activation(output: object) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Unable to extract a tensor activation from {type(output).__name__}")


def _load_image(path: str | Path, image_size: int) -> tuple[torch.Tensor, np.ndarray]:
    image = Image.open(path).convert("RGB")
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    height, width = image_np.shape[:2]
    if height == 0 or width == 0:
        raise ValueError(f"Invalid image size for {path!s}: {(height, width)}")

    scale = float(image_size) / float(max(height, width))
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    resized = np.asarray(
        image.resize((new_width, new_height), Image.Resampling.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    canvas = np.zeros((image_size, image_size, 3), dtype=np.float32)
    top = (image_size - new_height) // 2
    left = (image_size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized

    tensor = torch.from_numpy(canvas).permute(2, 0, 1)
    return tensor, canvas


def _build_baselines(batch: torch.Tensor, image_arrays: list[np.ndarray], baseline_mode: str) -> torch.Tensor:
    if baseline_mode == "zero":
        return torch.zeros_like(batch)
    if baseline_mode == "mean_rgb":
        baselines = []
        for image_np in image_arrays:
            mean_rgb = image_np.reshape(-1, 3).mean(axis=0, dtype=np.float32)
            baseline = np.broadcast_to(mean_rgb.reshape(1, 1, 3), image_np.shape).copy()
            baselines.append(torch.from_numpy(baseline).permute(2, 0, 1))
        return torch.stack(baselines, dim=0).to(device=batch.device, dtype=batch.dtype)
    if baseline_mode == "blur":
        baselines = []
        for image_np in image_arrays:
            image = Image.fromarray(np.asarray(np.round(image_np * 255.0), dtype=np.uint8))
            blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=16.0)), dtype=np.float32) / 255.0
            baselines.append(torch.from_numpy(blurred).permute(2, 0, 1))
        return torch.stack(baselines, dim=0).to(device=batch.device, dtype=batch.dtype)
    raise ValueError(f"Unsupported baseline_mode: {baseline_mode}")


def _load_batch(
    image_paths: list[str | Path],
    *,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[np.ndarray], list[str]]:
    tensors = []
    arrays: list[np.ndarray] = []
    sample_ids: list[str] = []
    seen: dict[str, int] = {}
    for index, path in enumerate(image_paths):
        tensor, array = _load_image(path, image_size)
        tensors.append(tensor)
        arrays.append(array)
        stem = Path(path).stem
        count = seen.get(stem, 0)
        seen[stem] = count + 1
        sample_ids.append(stem if count == 0 else f"{stem}_{count}")
    batch = torch.stack(tensors, dim=0).to(device=device, dtype=torch.float32)
    return batch, arrays, sample_ids


def _load_model(weights_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[int, str]]:
    yolo = YOLO(str(weights_path))
    model = yolo.model.to(device).eval()
    return model, dict(yolo.names)


def _forward_with_activation(
    model: torch.nn.Module,
    hook: LayerHook,
    inputs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    hook.clear()
    outputs = model(inputs)
    activation = _unwrap_activation(hook.get())
    _, logits = _split_classifier_output(outputs)
    return logits, activation


def _project_linf(
    candidate: torch.Tensor,
    *,
    center: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    delta = torch.clamp(candidate - center, min=-epsilon, max=epsilon)
    return torch.clamp(center + delta, min=0.0, max=1.0)


def _gather_target_probabilities(logits: torch.Tensor, target_classes: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    batch_index = torch.arange(logits.shape[0], device=logits.device)
    return probabilities[batch_index, target_classes]


def _reshape_flat_like_activation(flat_values: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """Reshape flattened neuron scores back to the corresponding activation tensor layout."""

    return flat_values.reshape(activation.shape[0], *activation.shape[1:])


def _importance_maps_from_flat_attr(
    flat_attr: torch.Tensor,
    activation: torch.Tensor,
    *,
    out_hw: tuple[int, int],
) -> list[np.ndarray]:
    """Project per-neuron signed attributions into image-aligned 2D importance maps."""

    attr_tensor = _reshape_flat_like_activation(flat_attr, activation)
    return [project_importance_map(attr_tensor[index : index + 1], out_hw) for index in range(attr_tensor.shape[0])]


def _estimate_sparse_selection_for_inputs(
    *,
    model: torch.nn.Module,
    hook: LayerHook,
    cpu_model: torch.nn.Module | None,
    cpu_hook: LayerHook | None,
    use_cpu_attribution: bool,
    inputs: torch.Tensor,
    baselines: torch.Tensor,
    target_classes: torch.Tensor,
    config: AttackConfig,
    device: torch.device,
):
    """Run cheap-IG selection either on the active device or on CPU fallback."""

    if use_cpu_attribution and cpu_model is not None and cpu_hook is not None:
        selection = estimate_sparse_selection(
            model=cpu_model,
            hook=cpu_hook,
            inputs=inputs.cpu(),
            baselines=baselines.cpu(),
            target_classes=target_classes.cpu(),
            config=config,
            split_output=_split_classifier_output,
            unwrap_activation=_unwrap_activation,
        )
        selection.positive_mask = selection.positive_mask.to(device=device)
        selection.negative_mask = selection.negative_mask.to(device=device)
        selection.detached_ia = selection.detached_ia.to(device=device)
        selection.detached_attr = selection.detached_attr.to(device=device)
        selection.tail_scalar = selection.tail_scalar.to(device=device)
        return selection

    return estimate_sparse_selection(
        model=model,
        hook=hook,
        inputs=inputs,
        baselines=baselines,
        target_classes=target_classes,
        config=config,
        split_output=_split_classifier_output,
        unwrap_activation=_unwrap_activation,
    )


def _write_attack_images(
    *,
    config: AttackConfig,
    sample_ids: list[str],
    current_images: torch.Tensor,
    best_images: torch.Tensor,
) -> tuple[list[str], list[str]]:
    latest_paths: list[str] = []
    best_paths: list[str] = []

    if current_images.shape[0] == 1:
        latest_path = config.latest_dir / "latest_adv.png"
        best_path = config.best_dir / "best_adv.png"
        if config.save_latest:
            save_image_tensor(current_images[0], latest_path)
        if config.save_best:
            save_image_tensor(best_images[0], best_path)
        latest_paths.append(str(latest_path))
        best_paths.append(str(best_path))
        return latest_paths, best_paths

    for index, sample_id in enumerate(sample_ids):
        latest_path = config.latest_dir / f"{sample_id}.png"
        best_path = config.best_dir / f"{sample_id}.png"
        if config.save_latest:
            save_image_tensor(current_images[index], latest_path)
        if config.save_best:
            save_image_tensor(best_images[index], best_path)
        latest_paths.append(str(latest_path))
        best_paths.append(str(best_path))
    return latest_paths, best_paths


def _run_single_batch(image_paths: list[str], config: AttackConfig) -> AttackRunResult:
    run_start = time.perf_counter()
    attribution_runtime_seconds = 0.0
    device = _resolve_device(config.force_device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.latest_dir.mkdir(parents=True, exist_ok=True)
    config.best_dir.mkdir(parents=True, exist_ok=True)

    model, class_names = _load_model(config.weights_path, device)
    hook = LayerHook(model, config.layer_name)
    cpu_model = None
    cpu_hook = None
    use_cpu_attribution = config.fallback_to_cpu_for_attribution and device.type != "cpu"

    if use_cpu_attribution:
        cpu_model, _ = _load_model(config.weights_path, torch.device("cpu"))
        cpu_hook = LayerHook(cpu_model, config.layer_name)

    try:
        clean_batch, image_arrays, sample_ids = _load_batch(
            image_paths,
            image_size=config.image_size,
            device=device,
        )
        baselines = _build_baselines(clean_batch, image_arrays, config.baseline_mode)

        with torch.no_grad():
            clean_logits, baseline_activation = _forward_with_activation(model, hook, clean_batch)
            baseline_logits, baseline_feature = _forward_with_activation(model, hook, baselines)
            del baseline_logits
            baseline_feature = baseline_feature.detach()

        target_classes = clean_logits.argmax(dim=1)
        clean_target_logits = clean_logits.gather(1, target_classes[:, None]).squeeze(1).detach()
        clean_target_probabilities = _gather_target_probabilities(clean_logits, target_classes).detach()

        x_adv = clean_batch.detach().clone()
        best_images = x_adv.detach().clone()
        best_drops = torch.zeros(x_adv.shape[0], device=device, dtype=x_adv.dtype)
        best_steps = torch.zeros(x_adv.shape[0], device=device, dtype=torch.long)
        momentum_buffer = torch.zeros_like(x_adv)

        history: list[dict[str, Any]] = []
        visualizer = LiveAttackVisualizer(
            clean_images=clean_batch.detach(),
            sample_ids=sample_ids,
            enabled=config.live_plots,
            method_label="Cheap-IG",
        )
        clean_importance_maps: list[np.ndarray] | None = None
        current_importance_maps: list[np.ndarray] | None = None
        cached_selection = None

        latest_paths: list[str] = []
        best_paths: list[str] = []
        batch_indices = torch.arange(x_adv.shape[0], device=device)

        for step in range(1, config.attack_steps + 1):
            if cached_selection is None:
                selection_start = time.perf_counter()
                selection = _estimate_sparse_selection_for_inputs(
                    model=model,
                    hook=hook,
                    cpu_model=cpu_model,
                    cpu_hook=cpu_hook,
                    use_cpu_attribution=use_cpu_attribution,
                    inputs=x_adv.detach(),
                    baselines=baselines.detach(),
                    target_classes=target_classes,
                    config=config,
                    device=device,
                )
                attribution_runtime_seconds += time.perf_counter() - selection_start
            else:
                selection = cached_selection
                cached_selection = None

            if clean_importance_maps is None:
                clean_importance_maps = _importance_maps_from_flat_attr(
                    selection.detached_attr,
                    baseline_activation,
                    out_hw=tuple(clean_batch.shape[-2:]),
                )

            x_adv_var = x_adv.detach().clone().requires_grad_(True)
            transformed_inputs = apply_dim(
                x_adv_var,
                probability=config.dim_prob,
                resize_low=config.dim_resize_low,
                resize_high=config.dim_resize_high,
            ) if config.variant == "pd" else x_adv_var

            logits_adv, activation_adv = _forward_with_activation(model, hook, transformed_inputs)
            attack_loss, current_surrogate_attr = sparse_weighted_surrogate_loss(
                current_activation=activation_adv,
                baseline_activation=baseline_feature,
                selection=selection,
                gamma=config.gamma,
            )
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
                best_images = torch.where(
                    improved[:, None, None, None],
                    x_adv.detach(),
                    best_images,
                )

                latest_paths, best_paths = _write_attack_images(
                    config=config,
                    sample_ids=sample_ids,
                    current_images=x_adv.detach(),
                    best_images=best_images.detach(),
                )

            selection_start = time.perf_counter()
            current_selection = _estimate_sparse_selection_for_inputs(
                model=model,
                hook=hook,
                cpu_model=cpu_model,
                cpu_hook=cpu_hook,
                use_cpu_attribution=use_cpu_attribution,
                inputs=x_adv.detach(),
                baselines=baselines.detach(),
                target_classes=target_classes,
                config=config,
                device=device,
            )
            attribution_runtime_seconds += time.perf_counter() - selection_start
            cached_selection = current_selection
            current_importance_maps = _importance_maps_from_flat_attr(
                current_selection.detached_attr,
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
                "selected_positive_counts": current_selection.selected_positive_counts,
                "selected_negative_counts": current_selection.selected_negative_counts,
                "fill_betas": current_selection.fill_betas,
                "surrogate_attr_mean_abs": float(current_selection.detached_attr.abs().mean().item()),
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
                    f"[step {step:02d}/{config.attack_steps}] "
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
            method_name="Cheap-IG",
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
        if cpu_hook is not None:
            cpu_hook.remove()
        _clear_backend_cache()


def run_attack(image_paths: list[str], config: AttackConfig) -> AttackRunResult:
    """Run the cheap-IG NAA attack on one batch or a sequence of micro-batches."""

    if not image_paths:
        raise ValueError("image_paths must contain at least one image")

    effective_batch_size = config.effective_batch_size or len(image_paths)
    if effective_batch_size >= len(image_paths):
        return _run_single_batch(image_paths, config)

    chunk_payloads: list[dict[str, Any]] = []
    merged: dict[str, Any] = {
        "image_paths": [],
        "sample_ids": [],
        "target_classes": [],
        "target_names": [],
        "clean_target_logits": [],
        "clean_target_probabilities": [],
        "final_target_logits": [],
        "final_target_probabilities": [],
        "best_target_logits": [],
        "best_target_probabilities": [],
        "final_target_logit_drops": [],
        "best_target_logit_drops": [],
        "best_steps": [],
        "latest_image_paths": [],
        "best_image_paths": [],
    }
    device_name = None

    for start in range(0, len(image_paths), effective_batch_size):
        chunk = image_paths[start : start + effective_batch_size]
        chunk_result = _run_single_batch(chunk, config)
        chunk_payloads.append(chunk_result.to_dict())
        device_name = chunk_result.device
        for key in merged:
            merged[key].extend(getattr(chunk_result, key))

    combined = AttackRunResult(
        image_paths=merged["image_paths"],
        sample_ids=merged["sample_ids"],
        device=device_name or "unknown",
        target_classes=merged["target_classes"],
        target_names=merged["target_names"],
        clean_target_logits=merged["clean_target_logits"],
        clean_target_probabilities=merged["clean_target_probabilities"],
        final_target_logits=merged["final_target_logits"],
        final_target_probabilities=merged["final_target_probabilities"],
        best_target_logits=merged["best_target_logits"],
        best_target_probabilities=merged["best_target_probabilities"],
        final_target_logit_drops=merged["final_target_logit_drops"],
        best_target_logit_drops=merged["best_target_logit_drops"],
        best_steps=merged["best_steps"],
        latest_image_paths=merged["latest_image_paths"],
        best_image_paths=merged["best_image_paths"],
        history=[],
        aggregate_history={},
        config=config.to_dict(),
        method_name="Cheap-IG",
        total_runtime_seconds=sum(chunk["total_runtime_seconds"] for chunk in chunk_payloads),
        attribution_runtime_seconds=sum(chunk["attribution_runtime_seconds"] for chunk in chunk_payloads),
        chunk_results=chunk_payloads,
    )
    if config.save_history_json:
        write_json(config.output_dir / "attack_history.json", combined.to_dict())
    return combined
