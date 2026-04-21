from __future__ import annotations

"""Paper-style disconnected grid benchmark for classifier attribution methods."""

import hashlib
import json
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from modules import IG, NAA, cheap_ig
from modules import alpha_segment_benchmark as seg
from modules.method_timing_cache import current_device_label, image_signature, load_or_compute_cached_value

try:
    from scipy.ndimage import gaussian_filter

    SCIPY_NDIMAGE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime environment dependent.
    gaussian_filter = None
    SCIPY_NDIMAGE_IMPORT_ERROR = exc


DEFAULT_CACHE_ROOT = "output/disconnected_grid_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_FILENAME = "disconnected_grid_report.md"
DEFAULT_SUMMARY_JSON = "disconnected_grid_summary.json"
DEFAULT_SPLIT_LAYER_NAME = "model.6"
DEFAULT_EVALUATION_LAYERS = ("input", "model.6")
DEFAULT_SETTINGS = ("gridpg", "dipart", "difull")
DEFAULT_N_STEPS = 128
DEFAULT_GRID_SIZE = 2
DEFAULT_GRID_IMAGE_SIZE = 224
DEFAULT_PREVIEW_IMAGES = 5
DEFAULT_INPUT_SMOOTHING_KERNELS = (33, 65, 129)
DEFAULT_LAYER_SMOOTHING_KERNELS = (5, 9, 17)
DEFAULT_AGGATT_BINS = ((0, 2), (2, 5), (5, 50), (50, 95), (95, 98), (98, 100))


def classifier_method_spec(kind, name=None, **kwargs):
    spec = {"kind": str(kind)}
    spec.update(kwargs)
    spec["name"] = str(name) if name is not None else _default_method_name(spec)
    return spec


def default_classifier_method_specs(cheap_ig_variants=None):
    specs = [
        classifier_method_spec("ig", name="IG"),
        classifier_method_spec("naa", name="NAA"),
    ]
    for variant in cheap_ig_variants or ():
        specs.append(classifier_method_spec("cheap_ig", **dict(variant)))
    return specs


def benchmark_classifier_disconnected_grid(
    *,
    image_paths,
    method_specs,
    split_layer_name=DEFAULT_SPLIT_LAYER_NAME,
    evaluation_layers=DEFAULT_EVALUATION_LAYERS,
    settings=DEFAULT_SETTINGS,
    grid_size=DEFAULT_GRID_SIZE,
    n_steps=DEFAULT_N_STEPS,
    grid_image_size=DEFAULT_GRID_IMAGE_SIZE,
    preview_images=DEFAULT_PREVIEW_IMAGES,
    input_smoothing_kernels=DEFAULT_INPUT_SMOOTHING_KERNELS,
    layer_smoothing_kernels=DEFAULT_LAYER_SMOOTHING_KERNELS,
    aggatt_bins=DEFAULT_AGGATT_BINS,
    output_dir=DEFAULT_OUTPUT_DIR,
    cache_root=DEFAULT_CACHE_ROOT,
    save_output=True,
    target_dir=None,
    report_filename=DEFAULT_REPORT_FILENAME,
    summary_filename=DEFAULT_SUMMARY_JSON,
    fd_eps=1e-3,
    clear_every=8,
    refresh_core=False,
    refresh_methods=False,
    refresh_evaluations=False,
    verbose=False,
):
    _require_scipy_ndimage()

    image_paths = [str(Path(path)) for path in image_paths]
    if not image_paths:
        raise ValueError("image_paths must not be empty.")
    settings = _normalize_settings(settings)
    evaluation_layers = _normalize_evaluation_layers(evaluation_layers)
    method_specs = _normalize_method_specs(method_specs)
    grid_size = int(grid_size)
    grid_image_size = int(grid_image_size)
    if grid_size != 2:
        raise ValueError("Only 2x2 grids are currently supported.")
    if grid_image_size % grid_size != 0:
        raise ValueError("grid_image_size must be divisible by grid_size.")
    preview_images = int(max(0, min(int(preview_images), len(image_paths))))

    dataset_context = _build_dataset_context(image_paths)
    smoothing_kernels = {
        "input": _normalize_kernel_values(input_smoothing_kernels),
        "model.6": _normalize_kernel_values(layer_smoothing_kernels),
    }
    aggatt_bins = _normalize_aggatt_bins(aggatt_bins)

    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = seg._prepare_existing_output_dir(target_dir)
        else:
            run_name = (
                f"disconnected_grid_classifier_{seg._safe_slug(split_layer_name)}"
                f"_steps_{int(n_steps)}_images_{int(len(image_paths))}"
            )
            run_dir = seg._prepare_output_dir(output_dir, run_name)

    core_rows = []
    method_rows = []
    rows = []
    per_image = []

    for image_index, image_path in enumerate(image_paths):
        image_payload = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "settings": {},
        }
        for setting in settings:
            core_record = _load_core_record(
                image_path=image_path,
                image_index=image_index,
                setting=setting,
                split_layer_name=split_layer_name,
                grid_size=grid_size,
                grid_image_size=grid_image_size,
                dataset_context=dataset_context,
                cache_root=cache_root,
                refresh=refresh_core,
            )
            if core_record.get("error") is not None or core_record.get("value") is None:
                raise RuntimeError(
                    f"Failed to build disconnected-grid core for {image_path} [{setting}]: "
                    f"{core_record.get('error')} (cache={core_record.get('cache_path')})"
                )
            core_value = core_record["value"]
            core_rows.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "setting": setting,
                    "core_cache_path": core_record["cache_path"],
                    "core_from_cache": core_record["from_cache"],
                    "core_duration_s": core_record["duration_s"],
                    "core": core_value,
                }
            )
            setting_payload = {
                "core": core_value,
                "evaluation_layers": {},
            }
            for eval_layer in evaluation_layers:
                layer_payload = {"methods": {}}
                for method_spec in method_specs:
                    method_record = _load_method_record(
                        image_path=image_path,
                        setting=setting,
                        eval_layer=eval_layer,
                        split_layer_name=split_layer_name,
                        n_steps=n_steps,
                        method_spec=method_spec,
                        core_value=core_value,
                        cache_root=cache_root,
                        refresh=refresh_methods,
                        fd_eps=fd_eps,
                        clear_every=clear_every,
                    )
                    if method_record.get("error") is not None or method_record.get("value") is None:
                        raise RuntimeError(
                            f"Failed to compute method {method_spec['name']} for {image_path} [{setting}/{eval_layer}]: "
                            f"{method_record.get('error')} (cache={method_record.get('cache_path')})"
                        )
                    method_value = method_record["value"]
                    method_rows.append(
                        {
                            "image_path": image_path,
                            "image_name": Path(image_path).name,
                            "setting": setting,
                            "eval_layer": eval_layer,
                            "method_name": method_spec["name"],
                            "method_id": method_spec["id"],
                            "kind": method_spec["kind"],
                            "method_cache_path": method_record["cache_path"],
                            "method_from_cache": method_record["from_cache"],
                            "method_duration_s": method_record["duration_s"],
                            **method_value,
                        }
                    )
                    evaluation_record = _load_evaluation_record(
                        image_path=image_path,
                        setting=setting,
                        eval_layer=eval_layer,
                        split_layer_name=split_layer_name,
                        method_spec=method_spec,
                        method_value=method_value,
                        core_value=core_value,
                        smoothing_kernels=smoothing_kernels[eval_layer],
                        aggatt_bins=aggatt_bins,
                        cache_root=cache_root,
                        refresh=refresh_evaluations,
                    )
                    if evaluation_record.get("error") is not None or evaluation_record.get("value") is None:
                        raise RuntimeError(
                            f"Failed to evaluate method {method_spec['name']} for {image_path} [{setting}/{eval_layer}]: "
                            f"{evaluation_record.get('error')} (cache={evaluation_record.get('cache_path')})"
                        )
                    evaluation_rows = evaluation_record["value"]["rows"]
                    for row in evaluation_rows:
                        rows.append(
                            {
                                "image_path": image_path,
                                "image_name": Path(image_path).name,
                                "setting": setting,
                                "eval_layer": eval_layer,
                                "method_name": method_spec["name"],
                                "method_id": method_spec["id"],
                                "kind": method_spec["kind"],
                                "method_duration_s": method_record["duration_s"],
                                "evaluation_cache_path": evaluation_record["cache_path"],
                                "evaluation_from_cache": evaluation_record["from_cache"],
                                "evaluation_duration_s": evaluation_record["duration_s"],
                                "benchmark_duration_s": float(method_record["duration_s"] or 0.0)
                                + float(evaluation_record["duration_s"] or 0.0),
                                **row,
                            }
                        )
                    layer_payload["methods"][method_spec["name"]] = {
                        "method": method_value,
                        "evaluation": evaluation_record["value"],
                    }
                setting_payload["evaluation_layers"][eval_layer] = layer_payload
            image_payload["settings"][setting] = setting_payload
        per_image.append(image_payload)

    summary = _build_summary(
        rows=rows,
        core_rows=core_rows,
        method_rows=method_rows,
        method_specs=method_specs,
        settings=settings,
        evaluation_layers=evaluation_layers,
        split_layer_name=split_layer_name,
        n_steps=n_steps,
        grid_image_size=grid_image_size,
        grid_size=grid_size,
        smoothing_kernels=smoothing_kernels,
        aggatt_bins=aggatt_bins,
        cache_root=cache_root,
        preview_images=preview_images,
    )

    figures = {}
    report_md = _build_report_markdown(summary, figures={})
    report_path = None
    summary_path = None
    if save_output:
        figures = _render_and_save_report_figures(run_dir=run_dir, summary=summary, rows=rows)
        report_md = _build_report_markdown(summary, figures=figures)
        report_path = run_dir / report_filename
        report_path.write_text(report_md + "\n", encoding="utf-8")
        summary_path = run_dir / summary_filename
        summary_path.write_text(
            seg._pretty_json(
                {
                    "summary": summary,
                    "rows": rows,
                    "core_rows": core_rows,
                    "method_rows": method_rows,
                }
            ),
            encoding="utf-8",
        )

    return {
        "task": "classifier",
        "image_paths": image_paths,
        "method_specs": method_specs,
        "settings": list(settings),
        "evaluation_layers": list(evaluation_layers),
        "split_layer_name": str(split_layer_name),
        "n_steps": int(n_steps),
        "grid_size": int(grid_size),
        "grid_image_size": int(grid_image_size),
        "smoothing_kernels": {key: list(values) for key, values in smoothing_kernels.items()},
        "aggatt_bins": [list(bounds) for bounds in aggatt_bins],
        "rows": rows,
        "core_rows": core_rows,
        "method_rows": method_rows,
        "per_image": per_image,
        "summary": summary,
        "report_markdown": report_md,
        "report_path": str(report_path) if report_path is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "figures": figures,
        "output_dir": str(run_dir) if run_dir is not None else None,
        "cache_root": str(cache_root),
    }


def render_disconnected_grid_report(result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = _render_and_save_report_figures(run_dir=output_dir, summary=result["summary"], rows=result["rows"])
    report_md = _build_report_markdown(result["summary"], figures=figures)
    report_path = output_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text(report_md + "\n", encoding="utf-8")
    summary_path = output_dir / DEFAULT_SUMMARY_JSON
    summary_path.write_text(
        seg._pretty_json(
            {
                "summary": result["summary"],
                "rows": result["rows"],
                "core_rows": result["core_rows"],
                "method_rows": result["method_rows"],
            }
        ),
        encoding="utf-8",
    )
    return {
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "figures": figures,
        "output_dir": str(output_dir),
    }


class _DisconnectedGridWrapper(torch.nn.Module):
    def __init__(self, *, base_model, setting, split_layer_name, grid_image_size):
        super().__init__()
        if str(split_layer_name) != "model.6":
            raise ValueError("Only split_layer_name='model.6' is currently supported.")
        self.setting = str(setting)
        self.grid_image_size = int(grid_image_size)
        self.grid_size = 2
        self.cell_size = self.grid_image_size // self.grid_size
        self.input_identity = torch.nn.Identity()
        self.explain_identity = torch.nn.Identity()
        seq = base_model.model
        self.prefix = torch.nn.ModuleList([seq[idx] for idx in range(0, 7)])
        self.tail = torch.nn.ModuleList([seq[idx] for idx in range(7, 10)])
        self.classifier_head = seq[10]

    def forward(self, x):
        x = self.input_identity(x)
        if self.setting == "gridpg":
            act = self._run_prefix(x)
            act = self.explain_identity(act)
            return self._classify(act, local_pool=False)

        cells = self._split_grid(x)
        if self.setting == "dipart":
            cell_acts = [self._run_prefix(cell) for cell in cells]
            act = self._tile_activations(cell_acts)
            act = self.explain_identity(act)
            return self._classify(act, local_pool=True)

        if self.setting == "difull":
            top_left_act = self._run_prefix(cells[0])
            act = self._embed_top_left(top_left_act)
            act = self.explain_identity(act)
            return self._classify(act, local_pool=True)

        raise ValueError(f"Unsupported setting: {self.setting}")

    def _run_prefix(self, x):
        value = x
        for module in self.prefix:
            value = module(value)
        return value

    def _run_tail(self, activation):
        value = activation
        for module in self.tail:
            value = module(value)
        return value

    def _classify(self, activation, *, local_pool):
        value = self._run_tail(activation)
        value = self.classifier_head.conv(value)
        if local_pool:
            h, w = value.shape[-2:]
            h_mid = int(ceil(h / 2.0))
            w_mid = int(ceil(w / 2.0))
            value = value[..., :h_mid, :w_mid]
        pooled = self.classifier_head.pool(value)
        pooled = self.classifier_head.drop(pooled)
        logits = self.classifier_head.linear(pooled.flatten(1))
        probs = torch.softmax(logits, dim=1)
        return probs, logits

    def _split_grid(self, x):
        size = self.cell_size
        return (
            x[..., :size, :size],
            x[..., :size, size:],
            x[..., size:, :size],
            x[..., size:, size:],
        )

    def _tile_activations(self, cell_acts):
        top = torch.cat([cell_acts[0], cell_acts[1]], dim=-1)
        bottom = torch.cat([cell_acts[2], cell_acts[3]], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def _embed_top_left(self, activation):
        batch, channels, height, width = activation.shape
        out = activation.new_zeros((batch, channels, height * 2, width * 2))
        out[..., :height, :width] = activation
        return out


def _load_core_record(
    *,
    image_path,
    image_index,
    setting,
    split_layer_name,
    grid_size,
    grid_image_size,
    dataset_context,
    cache_root,
    refresh,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "setting": str(setting),
        "split_layer_name": str(split_layer_name),
        "grid_size": int(grid_size),
        "grid_image_size": int(grid_image_size),
        "dataset_signature": dataset_context["dataset_signature"],
        "schema": 1,
    }
    sidecar_key = seg._config_hash(config)
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="disconnected_grid_core",
        config=config,
        label=f"{Path(image_path).stem}_{setting}",
        compute_fn=lambda: _compute_core_value(
            image_path=image_path,
            image_index=image_index,
            setting=setting,
            grid_image_size=grid_image_size,
            dataset_context=dataset_context,
            cache_root=cache_root,
            sidecar_key=sidecar_key,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _load_method_record(
    *,
    image_path,
    setting,
    eval_layer,
    split_layer_name,
    n_steps,
    method_spec,
    core_value,
    cache_root,
    refresh,
    fd_eps,
    clear_every,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "setting": str(setting),
        "eval_layer": str(eval_layer),
        "split_layer_name": str(split_layer_name),
        "n_steps": int(n_steps),
        "method_spec": method_spec,
        "target_class": int(core_value["target_class"]),
        "schema": 1,
    }
    sidecar_key = seg._config_hash(config)
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="disconnected_grid_method",
        config=config,
        label=f"{Path(image_path).stem}_{setting}_{seg._safe_slug(eval_layer)}_{method_spec['id']}",
        compute_fn=lambda: _compute_method_value(
            core_value=core_value,
            setting=setting,
            eval_layer=eval_layer,
            n_steps=n_steps,
            method_spec=method_spec,
            cache_root=cache_root,
            sidecar_key=sidecar_key,
            fd_eps=fd_eps,
            clear_every=clear_every,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _load_evaluation_record(
    *,
    image_path,
    setting,
    eval_layer,
    split_layer_name,
    method_spec,
    method_value,
    core_value,
    smoothing_kernels,
    aggatt_bins,
    cache_root,
    refresh,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "setting": str(setting),
        "eval_layer": str(eval_layer),
        "split_layer_name": str(split_layer_name),
        "method_spec": method_spec,
        "target_class": int(core_value["target_class"]),
        "smoothing_kernels": [int(v) for v in smoothing_kernels],
        "aggatt_bins": [list(v) for v in aggatt_bins],
        "schema": 1,
    }
    sidecar_key = seg._config_hash(config)
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="disconnected_grid_evaluation",
        config=config,
        label=f"{Path(image_path).stem}_{setting}_{seg._safe_slug(eval_layer)}_{method_spec['id']}",
        compute_fn=lambda: _compute_evaluation_value(
            method_value=method_value,
            core_value=core_value,
            cache_root=cache_root,
            sidecar_key=sidecar_key,
            smoothing_kernels=smoothing_kernels,
            aggatt_bins=aggatt_bins,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _compute_core_value(*, image_path, image_index, setting, grid_image_size, dataset_context, cache_root, sidecar_key):
    selection = _select_grid_sources(
        image_path=image_path,
        image_index=image_index,
        setting=setting,
        dataset_context=dataset_context,
    )
    grid_np = _compose_grid_image(selection["cell_paths"], grid_image_size=grid_image_size)
    grid_png_path = _png_sidecar_path(
        cache_root=cache_root,
        namespace="disconnected_grid_core",
        label=f"{Path(image_path).stem}_{setting}",
        sidecar_key=sidecar_key,
        kind="grid_image",
    )
    _save_png_sidecar(grid_png_path, grid_np)
    grid_array_path = seg._array_sidecar_path(
        cache_root=cache_root,
        namespace="disconnected_grid_core",
        label=f"{Path(image_path).stem}_{setting}",
        sidecar_key=sidecar_key,
        kind="grid_array",
    )
    seg._save_array_sidecar(grid_array_path, grid_np)

    x_target, _ = IG.load_image(image_path)
    with torch.no_grad():
        out = IG.model(x_target)
        _, logits = IG.split_classifier_output(out)
        target_class = int(logits[0].argmax().item())
        target_name = IG.class_names[target_class]
        target_logit = float(logits[0, target_class].item())

    cell_size = int(grid_image_size) // 2
    return {
        "image_path": str(image_path),
        "image_name": Path(image_path).name,
        "setting": str(setting),
        "target_class_name": selection["target_class_name"],
        "cell_paths": [str(path) for path in selection["cell_paths"]],
        "cell_class_names": [str(value) for value in selection["cell_class_names"]],
        "target_class": int(target_class),
        "target_name": str(target_name),
        "target_logit": float(target_logit),
        "grid_image_path": str(grid_png_path),
        "grid_array_path": str(grid_array_path),
        "grid_image_size": int(grid_image_size),
        "cell_size": int(cell_size),
        "target_cell": {"row": 0, "col": 0, "y0": 0, "y1": int(cell_size), "x0": 0, "x1": int(cell_size)},
    }


def _compute_method_value(*, core_value, setting, eval_layer, n_steps, method_spec, cache_root, sidecar_key, fd_eps, clear_every):
    _clear_all_backend_caches()
    grid_np = seg._load_array_sidecar(core_value["grid_array_path"]).astype(np.float32, copy=False)
    x = torch.from_numpy(grid_np).permute(2, 0, 1).unsqueeze(0).to(device=IG.DEVICE, dtype=IG.DTYPE)
    x0 = torch.zeros_like(x)
    wrapper = _DisconnectedGridWrapper(
        base_model=IG.model,
        setting=setting,
        split_layer_name="model.6",
        grid_image_size=int(core_value["grid_image_size"]),
    ).to(device=IG.DEVICE).eval()
    hook_name = "input_identity" if eval_layer == "input" else "explain_identity"
    hook = IG.LayerHook(wrapper, hook_name)
    try:
        def forward_with_layer(x_in):
            hook.clear()
            out = wrapper(x_in)
            act = IG.unwrap_tensor(hook.get())
            return out, act

        with torch.no_grad():
            out, act = forward_with_layer(x)
            _, logits = IG.split_classifier_output(out)
            out_x0, _ = forward_with_layer(x0)
            _, logits_x0 = IG.split_classifier_output(out_x0)
            target_logit = float(logits[0, int(core_value["target_class"])].item())
            fx = float(logits[0, int(core_value["target_class"])].item())
            fx0 = float(logits_x0[0, int(core_value["target_class"])].item())

        if method_spec["kind"] == "ig":
            cond_tensor, segment_steps = seg.compute_segment_conductance(
                model=wrapper,
                hook=hook,
                x=x,
                x0=x0,
                target_class=int(core_value["target_class"]),
                n_steps=n_steps,
                segment_start=float(method_spec.get("segment_start", 0.0)),
                segment_end=float(method_spec.get("segment_end", 1.0)),
                fd_eps=fd_eps,
                clear_every=clear_every,
            )
        elif method_spec["kind"] == "naa":
            cond_tensor, _, _, segment_steps = seg.compute_segment_naa_attribution(
                model=wrapper,
                hook=hook,
                x=x,
                x0=x0,
                target_class=int(core_value["target_class"]),
                n_steps=n_steps,
                segment_start=float(method_spec.get("segment_start", 0.0)),
                segment_end=float(method_spec.get("segment_end", 1.0)),
                clear_every=clear_every,
            )
        elif method_spec["kind"] == "cheap_ig":
            cond_tensor, segment_steps, cheap_stats = _compute_cheap_ig_tensor(
                model=wrapper,
                hook=hook,
                x=x,
                x0=x0,
                target_class=int(core_value["target_class"]),
                n_steps=n_steps,
                act_shape=tuple(int(v) for v in act.shape[1:]),
                segment_start=float(method_spec.get("segment_start", 0.0)),
                segment_end=float(method_spec.get("segment_end", 0.1)),
                selection_mode=method_spec.get("selection_mode", "positive"),
                selection_top_k=int(method_spec.get("selection_top_k", 4000)),
                fill_mode=method_spec.get("fill_mode", "naa_scaled"),
                fill_rho=float(method_spec.get("fill_rho", 0.8)),
                clear_every=clear_every,
            )
        else:
            raise ValueError(f"Unsupported method kind: {method_spec['kind']}")

        filter_scores = IG.reduce_filter_scores(cond_tensor)
        layer_score = cond_tensor.sum()
        abs_error = abs((fx - fx0) - float(layer_score.item()))
        unit_scores = cond_tensor[0].detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)
        overlay = _project_overlay_map(
            cond_tensor=cond_tensor,
            image_shape=tuple(grid_np.shape[:2]),
        ).astype(np.float32, copy=False)
        label = f"{Path(core_value['image_path']).stem}_{setting}_{seg._safe_slug(eval_layer)}_{method_spec['id']}"
        unit_scores_path = seg._array_sidecar_path(
            cache_root=cache_root,
            namespace="disconnected_grid_method",
            label=label,
            sidecar_key=sidecar_key,
            kind="unit_scores",
        )
        overlay_map_path = seg._array_sidecar_path(
            cache_root=cache_root,
            namespace="disconnected_grid_method",
            label=label,
            sidecar_key=sidecar_key,
            kind="overlay_map",
        )
        seg._save_array_sidecar(unit_scores_path, unit_scores)
        seg._save_array_sidecar(overlay_map_path, overlay)
        value = {
            "image_path": str(core_value["image_path"]),
            "setting": str(setting),
            "eval_layer": str(eval_layer),
            "method_name": method_spec["name"],
            "method_id": method_spec["id"],
            "kind": method_spec["kind"],
            "target_class": int(core_value["target_class"]),
            "target_name": core_value["target_name"],
            "target_logit": float(target_logit),
            "fx": float(fx),
            "fx0": float(fx0),
            "abs_error": float(abs_error),
            "activation_shape": [int(v) for v in act.shape],
            "segment_start": float(method_spec.get("segment_start", 0.0)),
            "segment_end": float(method_spec.get("segment_end", 1.0)),
            "segment_steps": int(segment_steps),
            "unit_scores_path": str(unit_scores_path),
            "overlay_map_path": str(overlay_map_path),
            "overlay_shape": [int(v) for v in overlay.shape],
        }
        if method_spec["kind"] == "cheap_ig":
            value.update(cheap_stats)
        return value
    finally:
        hook.remove()
        _clear_all_backend_caches()


def _compute_evaluation_value(*, method_value, core_value, cache_root, sidecar_key, smoothing_kernels, aggatt_bins):
    raw_overlay = seg._load_array_sidecar(method_value["overlay_map_path"]).astype(np.float32, copy=False)
    label = (
        f"{Path(core_value['image_path']).stem}_{core_value['setting']}_"
        f"{seg._safe_slug(method_value['eval_layer'])}_{method_value['method_id']}"
    )
    variants = [("raw", None, raw_overlay)]
    for kernel_size in smoothing_kernels:
        smoothed = _apply_gaussian_smoothing(raw_overlay, kernel_size)
        variants.append((f"gaussian_k{int(kernel_size)}", int(kernel_size), smoothed))

    rows = []
    for variant_name, kernel_size, overlay in variants:
        overlay_path = seg._array_sidecar_path(
            cache_root=cache_root,
            namespace="disconnected_grid_evaluation",
            label=label,
            sidecar_key=sidecar_key,
            kind=variant_name,
        )
        seg._save_array_sidecar(overlay_path, overlay.astype(np.float32, copy=False))
        metrics = _localization_metrics(
            overlay_map=overlay,
            target_cell=core_value["target_cell"],
            image_size=int(core_value["grid_image_size"]),
        )
        rows.append(
            {
                "smoothing_name": variant_name,
                "kernel_size": kernel_size,
                "overlay_map_path": str(overlay_path),
                "overlay_shape": [int(v) for v in overlay.shape],
                "aggatt_bin_edges": [list(bounds) for bounds in aggatt_bins],
                **metrics,
            }
        )
    return {"rows": rows}


def _compute_cheap_ig_tensor(
    *,
    model,
    hook,
    x,
    x0,
    target_class,
    n_steps,
    act_shape,
    segment_start,
    segment_end,
    selection_mode,
    selection_top_k,
    fill_mode,
    fill_rho,
    clear_every,
):
    cheap_ig._validate_segment(n_steps, segment_start, segment_end)
    fill_mode = cheap_ig._normalize_fill_mode(fill_mode)
    x = x.contiguous()
    x0 = x0.contiguous()
    delta_x = (x - x0).contiguous()
    alphas = cheap_ig._alpha_grid(n_steps)
    segment_idx = cheap_ig._segment_index_mask(alphas, segment_start, segment_end)
    segment_index_set = {int(idx) for idx in np.asarray(segment_idx, dtype=np.int64).tolist()}

    def forward_with_layer(x_in):
        hook.clear()
        out = model(x_in)
        act = IG.unwrap_tensor(hook.get())
        return out, act

    with torch.no_grad():
        _, act_x = forward_with_layer(x)
        _, act_x0 = forward_with_layer(x0)
        _, logits = IG.split_classifier_output(model(x))
        _, logits_x0 = IG.split_classifier_output(model(x0))
        delta_y = (act_x - act_x0).detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False)
        fx = float(logits[0, target_class].item())
        fx0 = float(logits_x0[0, target_class].item())
        fx_delta = fx - fx0

    sum_a_full = None
    sum_ab_segment = None
    segment_steps = 0

    def layer_activation(inp):
        _, activation = forward_with_layer(inp)
        return activation

    for step_idx, alpha in enumerate(alphas):
        x_alpha = (x0 + float(alpha) * delta_x).contiguous().detach().requires_grad_(True)
        out, act = forward_with_layer(x_alpha)
        _, logits = IG.split_classifier_output(out)
        score = logits[0, target_class]
        grad_y = torch.autograd.grad(score, act, retain_graph=False, create_graph=False)[0]
        grad_flat = grad_y.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False)
        if sum_a_full is None:
            sum_a_full = np.zeros_like(grad_flat, dtype=np.float64)
            sum_ab_segment = np.zeros_like(grad_flat, dtype=np.float64)
        sum_a_full += grad_flat
        if step_idx in segment_index_set:
            hook.clear()
            _, jvp_out = IG.jvp(layer_activation, (x_alpha,), (delta_x,))
            dir_flat = jvp_out.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False)
            sum_ab_segment += grad_flat * dir_flat
            segment_steps += 1
            del jvp_out
        del x_alpha, out, act, logits, score, grad_y
        hook.clear()
        if clear_every > 0 and ((step_idx + 1) % int(clear_every) == 0):
            _clear_all_backend_caches()

    if sum_a_full is None or sum_ab_segment is None or segment_steps == 0:
        raise RuntimeError("cheap-ig failed to accumulate attribution statistics")

    approx_vector = (sum_a_full / len(alphas)) * delta_y
    exact_vector = sum_ab_segment / segment_steps
    sparse_mask = cheap_ig._selection_mask(approx_vector, selection_mode, selection_top_k)
    filled_vector, fill_stats = cheap_ig._hybrid_fill_vector(
        exact_vector,
        approx_vector,
        sparse_mask,
        selection_mode=selection_mode,
        fill_mode=fill_mode,
        fill_rho=fill_rho,
    )
    cond_tensor = cheap_ig._reshape_classifier_vector_to_tensor(filled_vector, act_shape)
    cheap_stats = {
        "selection_mode": str(selection_mode),
        "selection_top_k": int(selection_top_k),
        "selected_neurons": int(np.asarray(sparse_mask, dtype=bool).sum()),
        "fill_mode": str(fill_mode),
        "fill_rho": float(fill_rho),
        "fill_beta": float(fill_stats["fill_beta"]),
        "filled_neurons": int(fill_stats["filled_neurons"]),
        "cheap_score_sum": float(cond_tensor.sum().item()),
        "cheap_abs_error": float(abs(fx_delta - float(cond_tensor.sum().item()))),
    }
    return cond_tensor, int(segment_steps), cheap_stats


def _build_summary(
    *,
    rows,
    core_rows,
    method_rows,
    method_specs,
    settings,
    evaluation_layers,
    split_layer_name,
    n_steps,
    grid_image_size,
    grid_size,
    smoothing_kernels,
    aggatt_bins,
    cache_root,
    preview_images,
):
    summary_rows = []
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["setting"], row["eval_layer"], row["method_name"], row["smoothing_name"])].append(row)
    for (setting, eval_layer, method_name, smoothing_name), subset in sorted(grouped.items()):
        summary_rows.append(
            {
                "setting": setting,
                "eval_layer": eval_layer,
                "method_name": method_name,
                "smoothing_name": smoothing_name,
                "kernel_size": subset[0]["kernel_size"],
                "localization": seg._stats_record([row["localization_score"] for row in subset]),
                "inside_positive_mass": seg._stats_record([row["inside_positive_mass"] for row in subset]),
                "outside_positive_mass": seg._stats_record([row["outside_positive_mass"] for row in subset]),
                "outside_positive_ratio": seg._stats_record([row["outside_positive_ratio"] for row in subset]),
                "runtime_s": seg._stats_record([row["benchmark_duration_s"] for row in subset]),
            }
        )

    summary_index = {
        (row["setting"], row["eval_layer"], row["method_name"], row["smoothing_name"]): row
        for row in summary_rows
    }
    best_variant_rows = []
    best_variant_per_method = {}
    raw_variant_rows = []
    for setting in settings:
        for eval_layer in evaluation_layers:
            for method_spec in method_specs:
                method_name = method_spec["name"]
                raw_row = summary_index[(setting, eval_layer, method_name, "raw")]
                raw_variant_rows.append(raw_row)
                candidates = [
                    row
                    for row in summary_rows
                    if row["setting"] == setting and row["eval_layer"] == eval_layer and row["method_name"] == method_name
                ]
                best_row = max(
                    candidates,
                    key=lambda row: (
                        _safe_float(row["localization"]["mean"]),
                        1 if row["smoothing_name"] != "raw" else 0,
                        -int(row["kernel_size"] or 0),
                    ),
                )
                best_variant_rows.append(best_row)
                best_variant_per_method[(setting, eval_layer, method_name)] = best_row["smoothing_name"]

    pairwise = {}
    for setting in settings:
        pairwise[setting] = {}
        for eval_layer in evaluation_layers:
            method_to_rows = {}
            for method_spec in method_specs:
                method_name = method_spec["name"]
                smoothing_name = best_variant_per_method[(setting, eval_layer, method_name)]
                subset = [
                    row
                    for row in rows
                    if row["setting"] == setting
                    and row["eval_layer"] == eval_layer
                    and row["method_name"] == method_name
                    and row["smoothing_name"] == smoothing_name
                ]
                method_to_rows[method_name] = {row["image_name"]: row for row in subset}
            pairwise[setting][eval_layer] = {}
            for left in [spec["name"] for spec in method_specs]:
                pairwise[setting][eval_layer][left] = {}
                for right in [spec["name"] for spec in method_specs]:
                    common = sorted(set(method_to_rows[left]) & set(method_to_rows[right]))
                    if not common:
                        pairwise[setting][eval_layer][left][right] = float("nan")
                        continue
                    wins = []
                    for image_name in common:
                        left_score = float(method_to_rows[left][image_name]["localization_score"])
                        right_score = float(method_to_rows[right][image_name]["localization_score"])
                        wins.append(1.0 if left_score > right_score else 0.0)
                    pairwise[setting][eval_layer][left][right] = float(np.mean(np.asarray(wins, dtype=np.float64)))

    preview_rows = [row for row in core_rows[:preview_images]]
    return {
        "task": "classifier",
        "split_layer_name": str(split_layer_name),
        "n_steps": int(n_steps),
        "grid_size": int(grid_size),
        "grid_image_size": int(grid_image_size),
        "settings": list(settings),
        "evaluation_layers": list(evaluation_layers),
        "method_names": [spec["name"] for spec in method_specs],
        "smoothing_kernels": {key: [int(v) for v in values] for key, values in smoothing_kernels.items()},
        "aggatt_bins": [list(bounds) for bounds in aggatt_bins],
        "n_images": int(len({row["image_path"] for row in core_rows})),
        "cache_root": str(cache_root),
        "summary_rows": summary_rows,
        "raw_variant_rows": raw_variant_rows,
        "best_variant_rows": best_variant_rows,
        "best_variant_per_method": {
            f"{setting}|{eval_layer}|{method_name}": smoothing_name
            for (setting, eval_layer, method_name), smoothing_name in best_variant_per_method.items()
        },
        "pairwise_win_rates": pairwise,
        "preview_core_rows": preview_rows,
        "core_runtime_s": seg._stats_record([row["core_duration_s"] for row in core_rows]),
        "method_runtime_s": seg._stats_record([row["method_duration_s"] for row in method_rows]),
    }


def _render_and_save_report_figures(*, run_dir, summary, rows):
    figure_dir = Path(run_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for setting in summary["settings"]:
        for eval_layer in summary["evaluation_layers"]:
            slug = f"{seg._safe_slug(setting)}_{seg._safe_slug(eval_layer)}"
            figures[f"summary_{slug}"] = seg._save_figure(
                _plot_summary_for_combo(summary, setting, eval_layer),
                figure_dir / f"summary_{slug}.png",
            )
            figures[f"distribution_{slug}"] = seg._save_figure(
                _plot_distribution_for_combo(summary, rows, setting, eval_layer),
                figure_dir / f"distribution_{slug}.png",
            )
            figures[f"pairwise_{slug}"] = seg._save_figure(
                _plot_pairwise_for_combo(summary, setting, eval_layer),
                figure_dir / f"pairwise_{slug}.png",
            )
            figures[f"aggatt_raw_{slug}"] = seg._save_figure(
                _plot_aggatt(summary, rows, setting, eval_layer, variant_scope="raw"),
                figure_dir / f"aggatt_raw_{slug}.png",
            )
            figures[f"aggatt_best_{slug}"] = seg._save_figure(
                _plot_aggatt(summary, rows, setting, eval_layer, variant_scope="best"),
                figure_dir / f"aggatt_best_{slug}.png",
            )
    return figures


def _plot_summary_for_combo(summary, setting, eval_layer):
    method_names = summary["method_names"]
    raw_rows = {
        row["method_name"]: row
        for row in summary["raw_variant_rows"]
        if row["setting"] == setting and row["eval_layer"] == eval_layer
    }
    best_rows = {
        row["method_name"]: row
        for row in summary["best_variant_rows"]
        if row["setting"] == setting and row["eval_layer"] == eval_layer
    }
    y = np.arange(len(method_names))
    raw_values = [float(raw_rows[name]["localization"]["mean"]) for name in method_names]
    best_values = [float(best_rows[name]["localization"]["mean"]) for name in method_names]
    labels = []
    for name in method_names:
        best_name = best_rows[name]["smoothing_name"]
        labels.append(name if best_name == "raw" else f"{name}\n(best={best_name})")
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.75 * len(method_names))), constrained_layout=True)
    ax.barh(y + 0.18, best_values, height=0.34, color="tab:blue", alpha=0.72, label="best variant")
    ax.barh(y - 0.18, raw_values, height=0.34, color="tab:gray", alpha=0.6, label="raw")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean localization score")
    ax.set_title(f"{setting} | {eval_layer}: raw vs best localization")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="best")
    return fig


def _plot_distribution_for_combo(summary, rows, setting, eval_layer):
    method_names = summary["method_names"]
    best_map = {
        row["method_name"]: row["smoothing_name"]
        for row in summary["best_variant_rows"]
        if row["setting"] == setting and row["eval_layer"] == eval_layer
    }
    samples = []
    labels = []
    for method_name in method_names:
        subset = [
            row["localization_score"]
            for row in rows
            if row["setting"] == setting
            and row["eval_layer"] == eval_layer
            and row["method_name"] == method_name
            and row["smoothing_name"] == best_map[method_name]
        ]
        samples.append(np.asarray(subset, dtype=np.float64))
        labels.append(method_name)
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    ax.boxplot(samples, labels=labels, showfliers=False)
    ax.set_ylabel("Localization score")
    ax.set_title(f"{setting} | {eval_layer}: best-variant localization distribution")
    ax.grid(axis="y", alpha=0.3)
    return fig


def _plot_pairwise_for_combo(summary, setting, eval_layer):
    method_names = list(summary["method_names"])
    matrix = np.full((len(method_names), len(method_names)), np.nan, dtype=np.float64)
    pairwise = summary["pairwise_win_rates"][setting][eval_layer]
    for row_idx, left in enumerate(method_names):
        for col_idx, right in enumerate(method_names):
            matrix[row_idx, col_idx] = float(pairwise[left][right])
    fig, ax = plt.subplots(
        figsize=(2.2 + 0.75 * len(method_names), 2.2 + 0.65 * len(method_names)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=35, ha="right")
    ax.set_yticklabels(method_names)
    ax.set_title(f"{setting} | {eval_layer}: pairwise win-rate (best variants)")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(
                col_idx,
                row_idx,
                "n/a" if value != value else f"{value:.2f}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(row > col)")
    return fig


def _plot_aggatt(summary, rows, setting, eval_layer, *, variant_scope):
    method_names = summary["method_names"]
    bins = [tuple(bounds) for bounds in summary["aggatt_bins"]]
    selected_rows = []
    for method_name in method_names:
        if variant_scope == "raw":
            smoothing_name = "raw"
        else:
            key = f"{setting}|{eval_layer}|{method_name}"
            smoothing_name = summary["best_variant_per_method"][key]
        subset = [
            row
            for row in rows
            if row["setting"] == setting
            and row["eval_layer"] == eval_layer
            and row["method_name"] == method_name
            and row["smoothing_name"] == smoothing_name
        ]
        subset.sort(key=lambda row: float(row["localization_score"]), reverse=True)
        selected_rows.append((method_name, smoothing_name, subset))

    n_rows = len(method_names)
    n_cols = len(bins)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.2 * n_cols, 2.2 * n_rows + 0.8),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(f"{setting} | {eval_layer}: AggAtt ({variant_scope})", fontsize=14)
    for row_idx, (method_name, smoothing_name, subset) in enumerate(selected_rows):
        agg_maps = _aggregate_bins(subset, bins)
        for col_idx, ((start, end), agg_map) in enumerate(zip(bins, agg_maps)):
            ax = axes[row_idx, col_idx]
            ax.imshow(agg_map, cmap="seismic", vmin=-1.0, vmax=1.0, interpolation="nearest")
            _draw_grid_lines(ax, int(summary["grid_image_size"]))
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"{start}-{end}%", fontsize=9)
            if col_idx == 0:
                label = method_name if smoothing_name == "raw" else f"{method_name}\n{smoothing_name}"
                ax.text(-0.05, 0.5, label, transform=ax.transAxes, va="center", ha="right", fontsize=8)
    return fig


def _build_report_markdown(summary, *, figures):
    lines = [
        "# Disconnected Grid Benchmark",
        "",
        "Classifier-only paper-style benchmark on synthetic `2x2` grids for `IG`, `NAA` and `Cheap-IG`.",
        "",
        "## Configuration",
        "",
        f"- split_layer_name=`{summary['split_layer_name']}`",
        f"- evaluation_layers=`{summary['evaluation_layers']}`",
        f"- settings=`{summary['settings']}`",
        f"- grid_size=`{summary['grid_size']}`",
        f"- grid_image_size=`{summary['grid_image_size']}`",
        f"- n_steps=`{summary['n_steps']}`",
        f"- n_images=`{summary['n_images']}`",
        f"- cache_root=`{summary['cache_root']}`",
        f"- smoothing_kernels=`{summary['smoothing_kernels']}`",
        "",
        "## Best Variant Summary",
        "",
        "| Setting | Layer | Method | Raw mean | Best variant | Best mean | Outside mean | runtime_s |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    raw_map = {
        (row["setting"], row["eval_layer"], row["method_name"]): row
        for row in summary["raw_variant_rows"]
    }
    for row in sorted(summary["best_variant_rows"], key=lambda item: (item["setting"], item["eval_layer"], item["method_name"])):
        raw_row = raw_map[(row["setting"], row["eval_layer"], row["method_name"])]
        lines.append(
            "| {setting} | {layer} | {method} | {raw_mean} | {variant} | {best_mean} | {outside} | {runtime} |".format(
                setting=row["setting"],
                layer=row["eval_layer"],
                method=row["method_name"],
                raw_mean=seg._format_number(raw_row["localization"]["mean"]),
                variant=row["smoothing_name"],
                best_mean=seg._format_number(row["localization"]["mean"]),
                outside=seg._format_number(row["outside_positive_mass"]["mean"]),
                runtime=seg._format_number(row["runtime_s"]["mean"]),
            )
        )

    for setting in summary["settings"]:
        for eval_layer in summary["evaluation_layers"]:
            slug = f"{seg._safe_slug(setting)}_{seg._safe_slug(eval_layer)}"
            lines.extend(
                [
                    "",
                    f"## {setting} / {eval_layer}",
                    "",
                    f"![]({seg._relative_markdown_path(figures[f'summary_{slug}'])})" if figures else "",
                    "",
                    f"![]({seg._relative_markdown_path(figures[f'distribution_{slug}'])})" if figures else "",
                    "",
                    f"![]({seg._relative_markdown_path(figures[f'pairwise_{slug}'])})" if figures else "",
                    "",
                    f"![]({seg._relative_markdown_path(figures[f'aggatt_raw_{slug}'])})" if figures else "",
                    "",
                    f"![]({seg._relative_markdown_path(figures[f'aggatt_best_{slug}'])})" if figures else "",
                ]
            )
    return "\n".join(line for line in lines if line is not None)


def _build_dataset_context(image_paths):
    image_paths = [str(Path(path)) for path in image_paths]
    class_to_paths = defaultdict(list)
    for path in image_paths:
        class_to_paths[_oxford_pet_class_name(path)].append(str(path))
    for paths in class_to_paths.values():
        paths.sort(key=lambda value: Path(value).name.lower())
    class_names = sorted(class_to_paths)
    dataset_signature = {
        "images": [image_signature(path) for path in image_paths],
        "class_names": class_names,
    }
    return {
        "image_paths": image_paths,
        "class_to_paths": dict(class_to_paths),
        "class_names": class_names,
        "dataset_signature": dataset_signature,
    }


def _select_grid_sources(*, image_path, image_index, setting, dataset_context):
    target_class_name = _oxford_pet_class_name(image_path)
    class_names = list(dataset_context["class_names"])
    class_to_paths = dataset_context["class_to_paths"]
    target_class_idx = class_names.index(target_class_name)
    distinct_classes = [name for name in class_names if name != target_class_name]
    if len(distinct_classes) < 3:
        raise ValueError(
            "Disconnected-grid benchmark requires at least 4 distinct classes in `image_paths` "
            f"to build `{setting}` grids. Got classes={class_names}."
        )
    start = int(image_index) % max(len(distinct_classes), 1)

    def pick_from_class(class_name, offset):
        paths = class_to_paths[class_name]
        return paths[(int(image_index) + int(offset)) % len(paths)]

    other_class_names = []
    other_paths = []
    for offset in range(3):
        class_name = distinct_classes[(start + offset) % len(distinct_classes)]
        other_class_names.append(class_name)
        other_paths.append(pick_from_class(class_name, offset))

    same_class_paths = class_to_paths[target_class_name]
    if len(same_class_paths) > 1:
        same_class_alt = same_class_paths[(same_class_paths.index(str(image_path)) + 1) % len(same_class_paths)]
    else:
        same_class_alt = str(image_path)

    if setting == "gridpg":
        cell_paths = [str(image_path), other_paths[0], other_paths[1], other_paths[2]]
        cell_class_names = [target_class_name, other_class_names[0], other_class_names[1], other_class_names[2]]
    else:
        cell_paths = [str(image_path), other_paths[0], other_paths[1], same_class_alt]
        cell_class_names = [target_class_name, other_class_names[0], other_class_names[1], target_class_name]
    return {
        "target_class_name": target_class_name,
        "cell_paths": cell_paths,
        "cell_class_names": cell_class_names,
        "target_class_idx": int(target_class_idx),
    }


def _oxford_pet_class_name(path):
    stem = Path(path).stem
    if "_" not in stem:
        return stem
    return stem.rsplit("_", 1)[0]


def _compose_grid_image(cell_paths, *, grid_image_size):
    cell_size = int(grid_image_size) // 2
    canvas = np.zeros((int(grid_image_size), int(grid_image_size), 3), dtype=np.float32)
    for idx, path in enumerate(cell_paths):
        row = idx // 2
        col = idx % 2
        cell = _load_letterboxed_rgb(path, cell_size)
        y0 = row * cell_size
        x0 = col * cell_size
        canvas[y0:y0 + cell_size, x0:x0 + cell_size] = cell
    return canvas


def _load_letterboxed_rgb(path, image_size):
    img = Image.open(path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]
    scale = float(image_size) / float(max(h, w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.asarray(
        Image.fromarray((img_np * 255.0).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0
    canvas = np.zeros((int(image_size), int(image_size), 3), dtype=np.float32)
    top = (int(image_size) - new_h) // 2
    left = (int(image_size) - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def _project_overlay_map(*, cond_tensor, image_shape):
    cond_np = cond_tensor[0].detach().cpu().numpy()
    if cond_np.ndim < 3:
        raise ValueError(f"Expected cond_tensor with [B,C,H,W], got {tuple(cond_tensor.shape)}")
    overlay = cond_np.sum(axis=0).astype(np.float32, copy=False)
    if tuple(overlay.shape) != tuple(image_shape):
        overlay = IG._resize_map_nearest(overlay, image_shape).astype(np.float32, copy=False)
    return overlay


def _apply_gaussian_smoothing(overlay_map, kernel_size):
    if kernel_size is None:
        return np.asarray(overlay_map, dtype=np.float32)
    sigma = float(kernel_size) / 4.0
    truncate = max(1.0, ((float(kernel_size) - 1.0) / 2.0) / max(sigma, 1e-12))
    return gaussian_filter(np.asarray(overlay_map, dtype=np.float32), sigma=sigma, truncate=truncate).astype(
        np.float32,
        copy=False,
    )


def _localization_metrics(*, overlay_map, target_cell, image_size, eps=1e-12):
    overlay = np.asarray(overlay_map, dtype=np.float64)
    positive = np.maximum(overlay, 0.0)
    y0 = int(target_cell["y0"])
    y1 = int(target_cell["y1"])
    x0 = int(target_cell["x0"])
    x1 = int(target_cell["x1"])
    inside = float(positive[y0:y1, x0:x1].sum())
    total = float(positive.sum())
    outside = float(max(total - inside, 0.0))
    localization = inside / max(total, eps) if total > eps else float("nan")
    return {
        "localization_score": float(localization),
        "inside_positive_mass": float(inside),
        "outside_positive_mass": float(outside),
        "outside_positive_ratio": float(outside / max(total, eps)) if total > eps else float("nan"),
    }


def _aggregate_bins(rows, bins):
    if not rows:
        return [np.zeros((DEFAULT_GRID_IMAGE_SIZE, DEFAULT_GRID_IMAGE_SIZE), dtype=np.float32) for _ in bins]
    agg_maps = []
    n = len(rows)
    for start_pct, end_pct in bins:
        start_idx = int(np.floor((float(start_pct) / 100.0) * n))
        end_idx = int(np.ceil((float(end_pct) / 100.0) * n))
        subset = rows[start_idx:end_idx]
        if not subset:
            subset = rows[max(0, min(start_idx, n - 1)): max(0, min(start_idx + 1, n))]
        maps = []
        for row in subset:
            overlay = seg._load_array_sidecar(row["overlay_map_path"])
            maps.append(seg._normalize_map(overlay))
        agg_maps.append(np.mean(np.asarray(maps, dtype=np.float32), axis=0).astype(np.float32, copy=False))
    return agg_maps


def _draw_grid_lines(ax, image_size):
    half = int(image_size) // 2
    ax.axhline(half - 0.5, color="white", linewidth=1.0, alpha=0.9)
    ax.axvline(half - 0.5, color="white", linewidth=1.0, alpha=0.9)


def _normalize_method_specs(method_specs):
    normalized = []
    seen_ids = set()
    for raw_spec in method_specs:
        spec = dict(raw_spec)
        spec["kind"] = str(spec["kind"])
        spec["name"] = str(spec.get("name") or _default_method_name(spec))
        spec["segment_start"] = float(spec.get("segment_start", 0.0))
        spec["segment_end"] = float(spec.get("segment_end", 1.0))
        spec["id"] = _method_id(spec)
        if spec["id"] in seen_ids:
            raise ValueError(f"Duplicate method spec id: {spec['id']}")
        seen_ids.add(spec["id"])
        normalized.append(spec)
    return normalized


def _default_method_name(spec):
    kind = str(spec["kind"])
    segment_start = float(spec.get("segment_start", 0.0))
    segment_end = float(spec.get("segment_end", 1.0))
    if kind == "ig":
        return f"IG[{segment_start:g},{segment_end:g}]"
    if kind == "naa":
        return f"NAA[{segment_start:g},{segment_end:g}]"
    selection_mode = spec.get("selection_mode", "positive")
    selection_top_k = int(spec.get("selection_top_k", 4000))
    fill_mode = str(spec.get("fill_mode", "naa_scaled"))
    if fill_mode == "zero":
        fill_fragment = "zero"
    else:
        fill_fragment = f"{fill_mode}/rho{float(spec.get('fill_rho', 0.8)):g}"
    return f"Cheap-IG[{segment_start:g},{segment_end:g}]/{selection_mode}/k{selection_top_k}/{fill_fragment}"


def _method_id(spec):
    material = json.dumps(spec, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.md5(material.encode("utf-8")).hexdigest()[:10]
    return f"{seg._safe_slug(spec['name'])}_{digest}"


def _normalize_settings(settings):
    allowed = {"gridpg", "dipart", "difull"}
    values = []
    for value in settings:
        value = str(value).lower()
        if value not in allowed:
            raise ValueError(f"Unsupported setting: {value}")
        if value not in values:
            values.append(value)
    return tuple(values)


def _normalize_evaluation_layers(evaluation_layers):
    allowed = {"input", "model.6"}
    values = []
    for value in evaluation_layers:
        value = str(value)
        if value not in allowed:
            raise ValueError(f"Unsupported evaluation layer: {value}")
        if value not in values:
            values.append(value)
    return tuple(values)


def _normalize_kernel_values(values):
    kernels = []
    for value in values:
        value = int(value)
        if value <= 0:
            raise ValueError(f"Kernel size must be positive, got {value}")
        if value % 2 == 0:
            raise ValueError(f"Kernel size must be odd, got {value}")
        if value not in kernels:
            kernels.append(value)
    return tuple(kernels)


def _normalize_aggatt_bins(bins):
    normalized = []
    for start, end in bins:
        start = float(start)
        end = float(end)
        if not 0.0 <= start <= end <= 100.0:
            raise ValueError(f"Invalid AggAtt bin: {(start, end)}")
        normalized.append((start, end))
    return tuple(normalized)


def _safe_float(value):
    value = float(value)
    return value if np.isfinite(value) else float("-inf")


def _png_sidecar_path(*, cache_root, namespace, label, sidecar_key, kind):
    root = Path(cache_root)
    filename = f"{seg._safe_slug(label)}_{seg._safe_slug(kind)}_{sidecar_key}.png"
    return root / namespace / "sidecars" / filename


def _save_png_sidecar(path, image_np):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(np.asarray(image_np) * 255.0, 0, 255).astype(np.uint8)).save(path)
    return str(path)


def _clear_all_backend_caches():
    for module in (IG, NAA):
        clear_fn = getattr(module, "clear_backend_cache", None)
        if callable(clear_fn):
            clear_fn()


def _require_scipy_ndimage():
    if gaussian_filter is None:
        raise ModuleNotFoundError(
            "scipy.ndimage is required for disconnected-grid smoothing, "
            f"but failed to import: {SCIPY_NDIMAGE_IMPORT_ERROR}"
        )
