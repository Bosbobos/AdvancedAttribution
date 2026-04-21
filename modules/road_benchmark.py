from __future__ import annotations

"""ROAD MoRF benchmark for classifier attribution methods."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import IG, NAA, cheap_ig
from modules.baseline_utils import (
    DEFAULT_BLUR_SIGMA,
    baseline_method_fragment,
    canonicalize_baseline_config,
)
from modules.method_timing_cache import current_device_label, image_signature, load_or_compute_cached_value

try:
    import scipy.sparse as scipy_sparse
    from scipy.sparse.linalg import spsolve as scipy_spsolve

    SCIPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime environment dependent.
    scipy_sparse = None
    scipy_spsolve = None
    SCIPY_IMPORT_ERROR = exc


DEFAULT_CACHE_ROOT = "output/road_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_FILENAME = "road_report.md"
DEFAULT_SUMMARY_JSON = "road_summary.json"
DEFAULT_PERCENTILES = (10, 20, 30, 40, 50, 60, 70, 80, 90)
DEFAULT_NLI_NOISE = 0.01
DEFAULT_NOISE_SEED = 0
DEFAULT_EPS = 1e-12
DEFAULT_PREVIEW_IMAGES = 5
DEFAULT_PREVIEW_GROUPS = (
    "ig_naa",
    "cheap_ig_zero",
    "cheap_ig_tail_rho08",
    "cheap_ig_tail_rho1",
)
_ORTHOGONAL_WEIGHT = 1.0 / 6.0
_DIAGONAL_WEIGHT = 1.0 / 12.0
NLI_NEIGHBOR_WEIGHTS = (
    ((-1, 0), _ORTHOGONAL_WEIGHT),
    ((1, 0), _ORTHOGONAL_WEIGHT),
    ((0, -1), _ORTHOGONAL_WEIGHT),
    ((0, 1), _ORTHOGONAL_WEIGHT),
    ((-1, -1), _DIAGONAL_WEIGHT),
    ((-1, 1), _DIAGONAL_WEIGHT),
    ((1, -1), _DIAGONAL_WEIGHT),
    ((1, 1), _DIAGONAL_WEIGHT),
)


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
        specs.append(classifier_method_spec("cheap_ig", **variant))
    return specs


def benchmark_classifier_road(
    *,
    image_paths,
    method_specs,
    layer_name,
    n_steps,
    percentiles=DEFAULT_PERCENTILES,
    noise=DEFAULT_NLI_NOISE,
    noise_seed=DEFAULT_NOISE_SEED,
    output_dir=DEFAULT_OUTPUT_DIR,
    cache_root=DEFAULT_CACHE_ROOT,
    save_output=True,
    target_dir=None,
    report_filename=DEFAULT_REPORT_FILENAME,
    summary_filename=DEFAULT_SUMMARY_JSON,
    top_n=0,
    fd_eps=1e-3,
    clear_every=8,
    refresh_core=False,
    refresh_methods=False,
    refresh_evaluations=False,
    verbose=False,
):
    _require_scipy()

    image_paths = [str(Path(path)) for path in image_paths]
    percentiles = _normalize_percentiles(percentiles)
    noise = _normalize_noise(noise)
    noise_seed = int(noise_seed)
    normalized_method_specs = _normalize_method_specs(method_specs)

    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = _prepare_existing_output_dir(target_dir)
        else:
            run_name = (
                f"road_classifier_morf_{_safe_slug(layer_name)}"
                f"_steps_{int(n_steps)}_images_{int(len(image_paths))}_pred_top1"
            )
            run_dir = _prepare_output_dir(output_dir, run_name)

    core_rows = []
    rows = []
    per_image = []

    for image_path in image_paths:
        core_record = _load_core_record(
            image_path=image_path,
            percentiles=percentiles,
            cache_root=cache_root,
            refresh=refresh_core,
        )
        if core_record.get("error") is not None or core_record.get("value") is None:
            raise RuntimeError(
                f"Failed to build ROAD core for {image_path}: {core_record.get('error')} "
                f"(cache={core_record.get('cache_path')})"
            )
        core_value = core_record["value"]
        core_rows.append(
            {
                "image_path": image_path,
                "core_cache_path": core_record["cache_path"],
                "core_from_cache": core_record["from_cache"],
                "core_duration_s": core_record["duration_s"],
                "core": core_value,
            }
        )

        image_payload = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "core": core_value,
            "methods": {},
        }

        for method_spec in normalized_method_specs:
            method_record = _load_method_record(
                image_path=image_path,
                layer_name=layer_name,
                n_steps=n_steps,
                method_spec=method_spec,
                cache_root=cache_root,
                refresh=refresh_methods,
                top_n=top_n,
                fd_eps=fd_eps,
                clear_every=clear_every,
            )
            if method_record.get("error") is not None or method_record.get("value") is None:
                raise RuntimeError(
                    f"Failed to compute method {method_spec['name']} for {image_path}: {method_record.get('error')} "
                    f"(cache={method_record.get('cache_path')})"
                )
            method_value = method_record["value"]

            evaluation_record = _load_evaluation_record(
                image_path=image_path,
                method_spec=method_spec,
                core_value=core_value,
                method_value=method_value,
                percentiles=percentiles,
                noise=noise,
                noise_seed=noise_seed,
                cache_root=cache_root,
                refresh=refresh_evaluations,
            )
            if evaluation_record.get("error") is not None or evaluation_record.get("value") is None:
                raise RuntimeError(
                    f"Failed to evaluate method {method_spec['name']} for {image_path}: {evaluation_record.get('error')} "
                    f"(cache={evaluation_record.get('cache_path')})"
                )
            evaluation = _ensure_evaluation_payload(evaluation_record["value"], core_value=core_value)
            row = {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "method_name": method_spec["name"],
                "method_id": method_spec["id"],
                "method_cache_path": method_record["cache_path"],
                "method_from_cache": method_record["from_cache"],
                "method_duration_s": method_record["duration_s"],
                "evaluation_cache_path": evaluation_record["cache_path"],
                "evaluation_from_cache": evaluation_record["from_cache"],
                "evaluation_duration_s": evaluation_record["duration_s"],
                "benchmark_duration_s": float(method_record["duration_s"] or 0.0)
                + float(evaluation_record["duration_s"] or 0.0),
                **evaluation,
            }
            rows.append(row)
            image_payload["methods"][method_spec["name"]] = {
                "cache_path": method_record["cache_path"],
                "from_cache": method_record["from_cache"],
                "duration_s": method_record["duration_s"],
                "evaluation_cache_path": evaluation_record["cache_path"],
                "evaluation_from_cache": evaluation_record["from_cache"],
                "evaluation_duration_s": evaluation_record["duration_s"],
                "method": method_value,
                "evaluation": evaluation,
            }

        per_image.append(image_payload)

    _attach_method_ranks(rows)
    summary = _build_benchmark_summary(
        rows=rows,
        core_rows=core_rows,
        method_specs=normalized_method_specs,
        layer_name=layer_name,
        n_steps=n_steps,
        percentiles=percentiles,
        noise=noise,
        noise_seed=noise_seed,
        cache_root=cache_root,
    )

    figures = {}
    preview_sections = []
    if run_dir is not None:
        figures = _render_and_save_report_figures(
            run_dir=run_dir,
            summary=summary,
            rows=rows,
            method_specs=normalized_method_specs,
        )
        preview_sections = _render_preview_sections(
            run_dir=run_dir,
            per_image=per_image,
            method_specs=normalized_method_specs,
        )
        report_md = _build_report_markdown(
            summary,
            rows,
            core_rows,
            method_specs=normalized_method_specs,
            figures=figures,
            preview_sections=preview_sections,
        )
        report_path = run_dir / report_filename
        report_path.write_text(report_md + "\n", encoding="utf-8")
        summary_path = run_dir / summary_filename
        summary_path.write_text(
            _pretty_json(
                {
                    "summary": summary,
                    "rows": rows,
                    "core_rows": core_rows,
                    "preview_sections": preview_sections,
                }
            ),
            encoding="utf-8",
        )
    else:
        report_md = _build_report_markdown(
            summary,
            rows,
            core_rows,
            method_specs=normalized_method_specs,
            figures={},
            preview_sections=[],
        )
        report_path = None
        summary_path = None

    return {
        "task": "classifier",
        "image_paths": image_paths,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "percentiles": [int(v) for v in percentiles],
        "noise": float(noise),
        "noise_seed": int(noise_seed),
        "cache_root": str(cache_root),
        "rows": rows,
        "core_rows": core_rows,
        "per_image": per_image,
        "method_specs": normalized_method_specs,
        "summary": summary,
        "preview_sections": preview_sections,
        "report_markdown": report_md,
        "report_path": str(report_path) if report_path is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "figures": figures,
        "output_dir": str(run_dir) if run_dir is not None else None,
    }


def _require_scipy():
    if scipy_sparse is None or scipy_spsolve is None:
        raise ModuleNotFoundError(
            "ROAD benchmark requires `scipy` for Noisy Linear Imputation. "
            "Install it with `python3 -m pip install scipy` and rerun the benchmark."
        ) from SCIPY_IMPORT_ERROR


def _load_core_record(*, image_path, percentiles, cache_root, refresh):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "percentiles": [int(v) for v in percentiles],
        "schema": 1,
    }
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="road_classifier_core",
        config=config,
        label=Path(image_path).stem,
        compute_fn=lambda: _compute_classifier_core_value(image_path=image_path, percentiles=percentiles),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _load_method_record(
    *,
    image_path,
    layer_name,
    n_steps,
    method_spec,
    cache_root,
    refresh,
    top_n,
    fd_eps,
    clear_every,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "method_spec": method_spec,
        "top_n": int(top_n),
        "fd_eps": float(fd_eps),
        "clear_every": int(clear_every),
        "schema": 1,
    }
    sidecar_key = _config_hash(config)
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="road_classifier_method",
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        compute_fn=lambda: _compute_classifier_method_value(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            method_spec=method_spec,
            cache_root=cache_root,
            sidecar_key=sidecar_key,
            top_n=top_n,
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
    method_spec,
    core_value,
    method_value,
    percentiles,
    noise,
    noise_seed,
    cache_root,
    refresh,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "method_spec": method_spec,
        "target_class": int(core_value["clean_top1_idx"]),
        "percentiles": [int(v) for v in percentiles],
        "noise": float(noise),
        "noise_seed": int(noise_seed),
        "schema": 1,
    }
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="road_classifier_evaluation",
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}_eval",
        compute_fn=lambda: _compute_classifier_evaluation_value(
            image_path=image_path,
            core_value=core_value,
            method_value=method_value,
            noise=noise,
            noise_seed=noise_seed,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _compute_classifier_core_value(*, image_path, percentiles):
    _clear_all_backend_caches()
    x, image_np = IG.load_image(image_path)
    try:
        with torch.no_grad():
            out = IG.model(x)
            _, logits = IG.split_classifier_output(out)
            probs = torch.softmax(logits[0], dim=0)
            top1_idx = int(logits[0].argmax().item())
            top1_name = IG.class_names[top1_idx]
            top1_logit = float(logits[0, top1_idx].item())
            top1_prob = float(probs[top1_idx].item())
        height, width = image_np.shape[:2]
        pixel_counts = _percentile_pixel_counts(height=height, width=width, percentiles=percentiles)
        return {
            "task": "classifier",
            "image_path": str(image_path),
            "clean_top1_idx": int(top1_idx),
            "clean_top1_name": top1_name,
            "clean_top1_logit": float(top1_logit),
            "clean_top1_prob": float(top1_prob),
            "image_height": int(height),
            "image_width": int(width),
            "n_pixels_total": int(height * width),
            "percentiles": [int(v) for v in percentiles],
            "removed_pixel_counts": [int(v) for v in pixel_counts],
        }
    finally:
        _clear_all_backend_caches()


def _compute_classifier_method_value(
    *,
    image_path,
    layer_name,
    n_steps,
    method_spec,
    cache_root,
    sidecar_key,
    top_n,
    fd_eps,
    clear_every,
):
    _clear_all_backend_caches()
    payload = _run_classifier_method(
        image_path=image_path,
        layer_name=layer_name,
        n_steps=n_steps,
        method_spec=method_spec,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
    )
    spatial_map = _project_spatial_map(payload)
    ranking_map, ranking_mode = _build_morf_ranking_map(spatial_map)
    spatial_path = _array_sidecar_path(
        cache_root=cache_root,
        namespace="road_classifier_method",
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        sidecar_key=sidecar_key,
        kind="spatial_map",
    )
    ranking_path = _array_sidecar_path(
        cache_root=cache_root,
        namespace="road_classifier_method",
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        sidecar_key=sidecar_key,
        kind="ranking_map",
    )
    _save_array_sidecar(spatial_path, spatial_map)
    _save_array_sidecar(ranking_path, ranking_map)
    _clear_all_backend_caches()
    return _serialize_method_payload(
        payload,
        method_spec,
        spatial_map_path=spatial_path,
        ranking_map_path=ranking_path,
        ranking_mode=ranking_mode,
        spatial_shape=spatial_map.shape,
    )


def _compute_classifier_evaluation_value(*, image_path, core_value, method_value, noise, noise_seed):
    _clear_all_backend_caches()
    runner = _ClassifierRoadRunner(image_path=image_path, target_class=core_value["clean_top1_idx"])
    try:
        imputer = NoisyLinearImputer(noise=noise)
        return _evaluate_method_with_runner(
            core_value,
            method_value,
            runner,
            imputer=imputer,
            noise_seed=noise_seed,
        )
    finally:
        runner.close()
        _clear_all_backend_caches()


def _run_classifier_method(*, image_path, layer_name, n_steps, method_spec, top_n, fd_eps, clear_every):
    kind = method_spec["kind"]
    common = {
        "image_path": image_path,
        "layer_name": layer_name,
        "n_steps": n_steps,
        "baseline_mode": method_spec.get("baseline_mode", "zero"),
        "baseline_rgb": method_spec.get("baseline_rgb"),
        "baseline_blur_sigma": method_spec.get("baseline_blur_sigma", DEFAULT_BLUR_SIGMA),
        "top_n": top_n,
        "clear_every": clear_every,
        "verbose": False,
        "show_total_plot": False,
        "show_filter_plots": False,
    }
    if kind == "ig":
        payload = IG.run_conductance_pipeline(fd_eps=fd_eps, **common)
    elif kind == "naa":
        payload = NAA.run_attribution_pipeline(**common)
    elif kind == "cheap_ig":
        payload = cheap_ig.run_classifier_cheap_ig_pipeline(
            segment_start=method_spec.get("segment_start", 0.0),
            segment_end=method_spec.get("segment_end", 0.1),
            selection_mode=method_spec.get("selection_mode", "signed"),
            selection_top_k=method_spec.get("selection_top_k", 5000),
            fill_mode=method_spec.get("fill_mode", "zero"),
            fill_rho=method_spec.get("fill_rho", 0.8),
            **common,
        )
    else:
        raise ValueError(f"Unsupported classifier method kind: {kind}")
    return payload


def _project_spatial_map(payload):
    cond_tensor = payload["cond_tensor"]
    cond_np = cond_tensor[0].detach().cpu().numpy()
    if cond_np.ndim < 3:
        raise ValueError(
            "ROAD benchmark requires a spatial attribution tensor with shape [B, C, H, W]. "
            f"Got cond_tensor shape={tuple(cond_tensor.shape)}."
        )
    spatial_map = cond_np.sum(axis=0).astype(np.float32, copy=False)
    image_np = payload["image_np"]
    if tuple(spatial_map.shape) != tuple(image_np.shape[:2]):
        spatial_map = IG._resize_map_nearest(spatial_map, image_np.shape[:2]).astype(np.float32, copy=False)
    return spatial_map


def _build_morf_ranking_map(spatial_map):
    ranking_map = np.maximum(np.asarray(spatial_map, dtype=np.float32), 0.0)
    if np.any(ranking_map > 0.0):
        return ranking_map, "positive_only"
    return np.abs(np.asarray(spatial_map, dtype=np.float32)), "abs_fallback"


def _serialize_method_payload(payload, method_spec, *, spatial_map_path, ranking_map_path, ranking_mode, spatial_shape):
    value = {
        "method_name": method_spec["name"],
        "method_id": method_spec["id"],
        "kind": method_spec["kind"],
        "target_class": int(payload.get("target_class", -1)),
        "target_name": payload.get("target_name"),
        "target_logit": float(payload.get("target_logit", float("nan"))),
        "target_prob": float(payload.get("target_prob", float("nan"))),
        "abs_error": float(payload.get("abs_error", float("nan"))),
        "fx": float(payload.get("fx", float("nan"))),
        "fx0": float(payload.get("fx0", float("nan"))),
        "spatial_map_path": str(spatial_map_path),
        "ranking_map_path": str(ranking_map_path),
        "ranking_mode": ranking_mode,
        "spatial_shape": [int(v) for v in spatial_shape],
    }
    for optional_key in (
        "selection_mode",
        "selection_top_k",
        "segment_start",
        "segment_end",
        "selected_neurons",
        "fill_mode",
        "fill_rho",
        "fill_beta",
        "filled_neurons",
        "baseline_mode",
        "baseline_rgb",
        "baseline_blur_sigma",
    ):
        if optional_key in payload:
            raw_value = payload.get(optional_key)
            if raw_value is None:
                value[optional_key] = None
            elif isinstance(raw_value, (np.integer, int)):
                value[optional_key] = int(raw_value)
            elif isinstance(raw_value, (np.floating, float)):
                value[optional_key] = float(raw_value)
            else:
                value[optional_key] = raw_value
    return value


class _ClassifierRoadRunner:
    def __init__(self, *, image_path, target_class=None):
        self.image_path = str(image_path)
        self.x, self.image_np = IG.load_image(self.image_path)
        with torch.no_grad():
            out = IG.model(self.x)
            _, logits = IG.split_classifier_output(out)
            probs = torch.softmax(logits[0], dim=0)
            self.clean_top1_idx = int(logits[0].argmax().item())
            self.clean_top1_name = IG.class_names[self.clean_top1_idx]
            self.clean_top1_logit = float(logits[0, self.clean_top1_idx].item())
            self.clean_top1_prob = float(probs[self.clean_top1_idx].item())
        self.target_class = int(self.clean_top1_idx if target_class is None else target_class)
        self.target_name = IG.class_names[self.target_class]
        self.clean_target_logit = float(logits[0, self.target_class].item())
        self.clean_target_prob = float(probs[self.target_class].item())
        self.height = int(self.image_np.shape[0])
        self.width = int(self.image_np.shape[1])
        self.n_pixels = int(self.height * self.width)

    def predict_from_image_np(self, image_np):
        x = torch.from_numpy(np.asarray(image_np, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0).to(IG.DEVICE, IG.DTYPE)
        with torch.no_grad():
            out = IG.model(x)
            _, logits = IG.split_classifier_output(out)
            probs = torch.softmax(logits[0], dim=0)
        top1_idx = int(logits[0].argmax().item())
        return {
            "top1_idx": int(top1_idx),
            "top1_name": IG.class_names[top1_idx],
            "top1_logit": float(logits[0, top1_idx].item()),
            "top1_prob": float(probs[top1_idx].item()),
            "target_logit": float(logits[0, self.target_class].item()),
            "target_prob": float(probs[self.target_class].item()),
        }

    def close(self):
        pass


class NoisyLinearImputer:
    """Sparse noisy linear imputation with fixed 8-neighborhood weights."""

    def __init__(self, noise=DEFAULT_NLI_NOISE):
        _require_scipy()
        self.noise = _normalize_noise(noise)

    def impute(self, image_np, remove_mask, *, seed):
        image_np = np.asarray(image_np, dtype=np.float32)
        remove_mask = np.asarray(remove_mask, dtype=bool)
        if image_np.ndim != 3:
            raise ValueError(f"Expected image_np with shape [H, W, C], got shape={image_np.shape}.")
        if remove_mask.shape != image_np.shape[:2]:
            raise ValueError(
                f"Mask shape mismatch: mask={remove_mask.shape}, image={image_np.shape[:2]}."
            )
        if not np.any(remove_mask):
            return image_np.copy()

        matrix, rhs, missing_indices = self._build_sparse_system(remove_mask, image_np)
        if matrix.shape[0] == 0:
            return image_np.copy()

        solved_channels = []
        for channel_idx in range(rhs.shape[1]):
            solved = scipy_spsolve(matrix, rhs[:, channel_idx])
            solved_channels.append(np.asarray(solved, dtype=np.float64).reshape(-1))
        solved_values = np.stack(solved_channels, axis=1)

        if self.noise > 0.0:
            rng = np.random.default_rng(int(seed))
            solved_values = solved_values + rng.normal(
                loc=0.0,
                scale=self.noise,
                size=solved_values.shape,
            )

        flat_image = image_np.reshape(-1, image_np.shape[2]).astype(np.float64, copy=True)
        flat_image[missing_indices] = np.clip(solved_values, 0.0, 1.0)
        return flat_image.reshape(image_np.shape).astype(np.float32, copy=False)

    def _build_sparse_system(self, remove_mask, image_np):
        height, width = remove_mask.shape
        flat_mask = remove_mask.reshape(-1)
        missing_indices = np.flatnonzero(flat_mask)
        coords_to_row = np.full(flat_mask.shape[0], -1, dtype=np.int64)
        coords_to_row[missing_indices] = np.arange(missing_indices.size, dtype=np.int64)
        image_flat = image_np.reshape(-1, image_np.shape[2]).astype(np.float64, copy=False)

        rows = []
        cols = []
        data = []
        rhs = np.zeros((missing_indices.size, image_np.shape[2]), dtype=np.float64)

        for row_idx, flat_idx in enumerate(missing_indices.tolist()):
            y, x = divmod(int(flat_idx), int(width))
            diag = 0.0
            for (dy, dx), weight in NLI_NEIGHBOR_WEIGHTS:
                ny = y + int(dy)
                nx = x + int(dx)
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                neighbor_flat = int(ny * width + nx)
                diag += float(weight)
                if flat_mask[neighbor_flat]:
                    rows.append(row_idx)
                    cols.append(int(coords_to_row[neighbor_flat]))
                    data.append(-float(weight))
                else:
                    rhs[row_idx, :] += float(weight) * image_flat[neighbor_flat, :]
            if diag <= DEFAULT_EPS:
                rows.append(row_idx)
                cols.append(row_idx)
                data.append(1.0)
            else:
                rows.append(row_idx)
                cols.append(row_idx)
                data.append(float(diag))

        matrix = scipy_sparse.csr_matrix(
            (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
            shape=(missing_indices.size, missing_indices.size),
        )
        return matrix, rhs, missing_indices


def _evaluate_method_with_runner(core_value, method_value, runner, *, imputer, noise_seed):
    ranking_map = _load_array_sidecar(method_value["ranking_map_path"])
    if tuple(ranking_map.shape) != (runner.height, runner.width):
        raise ValueError(
            f"Ranking map shape mismatch for {method_value['method_name']}: "
            f"{ranking_map.shape} vs {(runner.height, runner.width)}."
        )

    flat_scores = np.asarray(ranking_map, dtype=np.float64).reshape(-1)
    order = _stable_descending_order(flat_scores)
    counts = [int(v) for v in core_value["removed_pixel_counts"]]
    percentiles = [int(v) for v in core_value["percentiles"]]

    consistency_curve = []
    drop_curve = []
    target_logit_curve = []
    target_logit_drop_curve = []
    target_logit_drop_curve_normalized = []
    predicted_top1_indices = []
    predicted_top1_names = []

    for percentile, count in zip(percentiles, counts):
        remove_mask = np.zeros(flat_scores.shape[0], dtype=bool)
        remove_mask[order[:count]] = True
        remove_mask = remove_mask.reshape((runner.height, runner.width))
        imputed_image = imputer.impute(
            runner.image_np,
            remove_mask,
            seed=_derived_seed(noise_seed, runner.image_path, method_value["method_id"], percentile),
        )
        prediction = runner.predict_from_image_np(imputed_image)
        consistency = 1.0 if int(prediction["top1_idx"]) == int(runner.target_class) else 0.0
        target_logit_drop = float(runner.clean_target_logit - prediction["target_logit"])
        target_logit_drop_normalized = float(target_logit_drop / max(abs(runner.clean_target_logit), DEFAULT_EPS))
        consistency_curve.append(float(consistency))
        drop_curve.append(float(1.0 - consistency))
        target_logit_curve.append(float(prediction["target_logit"]))
        target_logit_drop_curve.append(target_logit_drop)
        target_logit_drop_curve_normalized.append(target_logit_drop_normalized)
        predicted_top1_indices.append(int(prediction["top1_idx"]))
        predicted_top1_names.append(prediction["top1_name"])

    mean_consistency = float(np.mean(consistency_curve)) if consistency_curve else float("nan")
    mean_drop = float(np.mean(drop_curve)) if drop_curve else float("nan")
    target_logit_drop_aoc = float(np.mean(target_logit_drop_curve)) if target_logit_drop_curve else float("nan")
    target_logit_drop_aoc_normalized = (
        float(np.mean(target_logit_drop_curve_normalized)) if target_logit_drop_curve_normalized else float("nan")
    )
    head_n = min(20, int(order.size))
    head_indices = [int(v) for v in order[:head_n].tolist()]
    head_coords = [list(np.unravel_index(idx, (runner.height, runner.width))) for idx in head_indices]
    return {
        "target_logit_drop_aoc": target_logit_drop_aoc,
        "target_logit_drop_aoc_normalized": target_logit_drop_aoc_normalized,
        "road_morf_mean_drop": mean_drop,
        "road_morf_mean_consistency": mean_consistency,
        "score": target_logit_drop_aoc,
        "consistency_curve": [float(v) for v in consistency_curve],
        "drop_curve": [float(v) for v in drop_curve],
        "target_logit_curve": [float(v) for v in target_logit_curve],
        "target_logit_drop_curve": [float(v) for v in target_logit_drop_curve],
        "target_logit_drop_curve_normalized": [float(v) for v in target_logit_drop_curve_normalized],
        "predicted_top1_indices": predicted_top1_indices,
        "predicted_top1_names": predicted_top1_names,
        "ordered_pixel_indices_head": head_indices,
        "ordered_pixel_coords_head": head_coords,
        "ranking_mode": method_value["ranking_mode"],
        "clean_target_logit": float(runner.clean_target_logit),
        "abs_error": float(method_value.get("abs_error", float("nan"))),
        "fx": float(method_value.get("fx", float("nan"))),
        "fx0": float(method_value.get("fx0", float("nan"))),
        "selected_neurons": method_value.get("selected_neurons"),
    }


def _ensure_evaluation_payload(evaluation, *, core_value):
    if evaluation is None:
        return None
    payload = dict(evaluation)
    clean_target_logit = float(payload.get("clean_target_logit", core_value.get("clean_top1_logit", float("nan"))))
    target_logit_curve = [float(v) for v in payload.get("target_logit_curve", [])]
    if "target_logit_drop_curve" not in payload and target_logit_curve:
        payload["target_logit_drop_curve"] = [float(clean_target_logit - float(v)) for v in target_logit_curve]
    if "target_logit_drop_curve_normalized" not in payload and payload.get("target_logit_drop_curve"):
        denom = max(abs(clean_target_logit), DEFAULT_EPS)
        payload["target_logit_drop_curve_normalized"] = [float(v) / denom for v in payload["target_logit_drop_curve"]]
    if "target_logit_drop_aoc" not in payload:
        if payload.get("target_logit_drop_curve"):
            payload["target_logit_drop_aoc"] = float(np.mean(np.asarray(payload["target_logit_drop_curve"], dtype=np.float64)))
        else:
            payload["target_logit_drop_aoc"] = float(payload.get("score", float("nan")))
    if "target_logit_drop_aoc_normalized" not in payload:
        if payload.get("target_logit_drop_curve_normalized"):
            payload["target_logit_drop_aoc_normalized"] = float(
                np.mean(np.asarray(payload["target_logit_drop_curve_normalized"], dtype=np.float64))
            )
        else:
            denom = max(abs(clean_target_logit), DEFAULT_EPS)
            payload["target_logit_drop_aoc_normalized"] = float(payload["target_logit_drop_aoc"]) / denom
    payload["clean_target_logit"] = clean_target_logit
    payload["score"] = float(payload.get("target_logit_drop_aoc", payload.get("score", float("nan"))))
    return payload


def _attach_method_ranks(rows):
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_path"], []).append(row)
    for image_rows in by_image.values():
        valid = [row for row in image_rows if row["score"] == row["score"]]
        valid.sort(key=lambda item: (-item["score"], item["method_name"]))
        rank_map = {row["method_name"]: index + 1 for index, row in enumerate(valid)}
        for row in image_rows:
            row["rank"] = rank_map.get(row["method_name"])


def _build_benchmark_summary(
    *,
    rows,
    core_rows,
    method_specs,
    layer_name,
    n_steps,
    percentiles,
    noise,
    noise_seed,
    cache_root,
):
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method_name"], []).append(row)

    method_summaries = {}
    for method_spec in method_specs:
        method_name = method_spec["name"]
        method_rows = by_method.get(method_name, [])
        method_summaries[method_name] = {
            "id": method_spec["id"],
            "kind": method_spec["kind"],
            "score": _stats_record([row.get("score") for row in method_rows]),
            "target_logit_drop_aoc": _stats_record([row.get("target_logit_drop_aoc") for row in method_rows]),
            "target_logit_drop_aoc_normalized": _stats_record(
                [row.get("target_logit_drop_aoc_normalized") for row in method_rows]
            ),
            "road_morf_mean_drop": _stats_record([row.get("road_morf_mean_drop") for row in method_rows]),
            "road_morf_mean_consistency": _stats_record([row.get("road_morf_mean_consistency") for row in method_rows]),
            "abs_error": _stats_record([row.get("abs_error") for row in method_rows]),
            "runtime_s": _stats_record([row.get("method_duration_s") for row in method_rows]),
            "evaluation_runtime_s": _stats_record([row.get("evaluation_duration_s") for row in method_rows]),
            "benchmark_runtime_s": _stats_record([row.get("benchmark_duration_s") for row in method_rows]),
            "rank": _stats_record([row.get("rank") for row in method_rows if row.get("rank") is not None]),
        }

    core_summary = {
        "clean_top1_logit": _stats_record([row["core"]["clean_top1_logit"] for row in core_rows]),
        "clean_top1_prob": _stats_record([row["core"]["clean_top1_prob"] for row in core_rows]),
        "n_pixels_total": _stats_record([row["core"]["n_pixels_total"] for row in core_rows]),
        "n_percentiles": _stats_record([len(row["core"]["percentiles"]) for row in core_rows]),
        "core_runtime_s": _stats_record([row["core_duration_s"] for row in core_rows]),
    }

    removed_pixel_counts = core_rows[0]["core"]["removed_pixel_counts"] if core_rows else []
    return {
        "task": "classifier",
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "n_images": int(len(core_rows)),
        "percentiles": [int(v) for v in percentiles],
        "removed_pixel_counts": [int(v) for v in removed_pixel_counts],
        "noise": float(noise),
        "noise_seed": int(noise_seed),
        "cache_root": str(cache_root),
        "method_summaries": method_summaries,
        "core_summary": core_summary,
        "pairwise_win_rate": _pairwise_win_rate(rows, method_specs),
    }


def _render_and_save_report_figures(*, run_dir, summary, rows, method_specs):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    fig = _plot_road_summary(summary, method_specs)
    path = figure_dir / "road_summary.png"
    _save_figure(fig, path)
    figures["road_summary"] = str(path)

    fig = _plot_road_distributions(rows, method_specs)
    path = figure_dir / "road_distributions.png"
    _save_figure(fig, path)
    figures["road_distributions"] = str(path)

    fig = _plot_road_curves(rows, summary["percentiles"], method_specs)
    path = figure_dir / "road_curves.png"
    _save_figure(fig, path)
    figures["road_curves"] = str(path)

    fig = _plot_road_family_envelope(rows, summary["percentiles"], method_specs)
    path = figure_dir / "road_family_envelope.png"
    _save_figure(fig, path)
    figures["road_family_envelope"] = str(path)

    fig = _plot_cheap_ig_heatmaps(summary, method_specs)
    path = figure_dir / "road_cheap_ig_heatmaps.png"
    _save_figure(fig, path)
    figures["road_cheap_ig_heatmaps"] = str(path)

    fig = _plot_cheap_ig_delta_boxplots(rows, method_specs)
    path = figure_dir / "road_cheap_ig_delta_boxplots.png"
    _save_figure(fig, path)
    figures["road_cheap_ig_delta_boxplots"] = str(path)

    fig = _plot_pairwise_win_heatmap(summary, method_specs)
    path = figure_dir / "road_pairwise_wins.png"
    _save_figure(fig, path)
    figures["road_pairwise_wins"] = str(path)

    return figures


def _render_preview_sections(*, run_dir, per_image, method_specs):
    preview_dir = run_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_images = per_image[:DEFAULT_PREVIEW_IMAGES]
    if not preview_images:
        return []

    rows = []
    method_index = {spec["name"]: spec for spec in method_specs}

    for image_idx, image_payload in enumerate(preview_images, start=1):
        _, image_np = IG.load_image(image_payload["image_path"])
        row = {
            "image_name": image_payload["image_name"],
        }
        raw_path = preview_dir / f"{image_idx:02d}_{_safe_slug(Path(image_payload['image_name']).stem)}_raw.png"
        if not raw_path.exists():
            fig = _build_raw_image_figure(image_np, title=image_payload["image_name"])
            _save_figure(fig, raw_path)
        row["raw_output_path"] = str(raw_path)

        for method_name, payload in image_payload["methods"].items():
            method_value = payload["method"]
            spatial_map = _load_array_sidecar(method_value["spatial_map_path"])
            fig = _build_spatial_overlay_figure(
                image_np=image_np,
                spatial_map=spatial_map,
                title=method_name,
            )
            output_path = preview_dir / (
                f"{image_idx:02d}_{_safe_slug(Path(image_payload['image_name']).stem)}_{method_index[method_name]['id']}.png"
            )
            _save_figure(fig, output_path)
            row[f"{method_index[method_name]['id']}_output_path"] = str(output_path)
        rows.append(row)

    sections = []
    for group_key in DEFAULT_PREVIEW_GROUPS:
        columns = _preview_columns_for_group(group_key, method_specs)
        if not columns:
            continue
        markdown = _build_preview_markdown_for_columns(rows, columns)
        if not markdown.strip():
            continue
        sections.append(
            {
                "group_key": group_key,
                "title": _preview_group_title(group_key),
                "columns": [dict(column) for column in columns],
                "markdown_table": markdown,
            }
        )
    return sections


def _preview_columns_for_group(group_key, method_specs):
    columns = [{"key": "raw", "title": "Raw"}]
    for spec in method_specs:
        if group_key == "ig_naa" and spec["kind"] in {"ig", "naa"}:
            columns.append({"key": spec["id"], "title": spec["name"]})
        elif group_key == "cheap_ig_zero" and spec["kind"] == "cheap_ig" and spec.get("fill_mode", "zero") == "zero":
            columns.append({"key": spec["id"], "title": spec["name"]})
        elif (
            group_key == "cheap_ig_tail_rho08"
            and spec["kind"] == "cheap_ig"
            and spec.get("fill_mode", "zero") == "naa_scaled"
            and abs(float(spec.get("fill_rho", 0.8)) - 0.8) <= DEFAULT_EPS
        ):
            columns.append({"key": spec["id"], "title": spec["name"]})
        elif (
            group_key == "cheap_ig_tail_rho1"
            and spec["kind"] == "cheap_ig"
            and spec.get("fill_mode", "zero") == "naa_scaled"
            and abs(float(spec.get("fill_rho", 0.8)) - 1.0) <= DEFAULT_EPS
        ):
            columns.append({"key": spec["id"], "title": spec["name"]})
    return columns if len(columns) > 1 else []


def _preview_group_title(group_key):
    mapping = {
        "ig_naa": "IG / NAA",
        "cheap_ig_zero": "Cheap-IG no tail",
        "cheap_ig_tail_rho08": "Cheap-IG + NAA tail (rho=0.8)",
        "cheap_ig_tail_rho1": "Cheap-IG + NAA tail (rho=1.0)",
    }
    return mapping[group_key]


def _build_report_markdown(summary, rows, core_rows, *, method_specs, figures, preview_sections):
    lines = [
        "# ROAD MoRF Benchmark",
        "",
        f"- task=`{summary['task']}`",
        f"- layer_name=`{summary['layer_name']}`",
        f"- n_steps={summary['n_steps']}",
        f"- n_images={summary['n_images']}",
        f"- percentiles={summary['percentiles']}",
        f"- noise={summary['noise']:.4f}",
        f"- noise_seed={summary['noise_seed']}",
        f"- cache_root=`{summary['cache_root']}`",
        "",
        "## Aggregate Summary",
        "",
        "| Method | Target logit drop AOC | Target logit drop AOC / |clean logit| | Top-1 mean drop | Mean consistency | Mean Rank | Attr Runtime (s) | Eval Runtime (s) | Abs Error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for method_spec in method_specs:
        stats = summary["method_summaries"][method_spec["name"]]
        lines.append(
            f"| {method_spec['name']} | "
            + " | ".join(
                [
                    _format_stats(stats["target_logit_drop_aoc"]),
                    _format_stats(stats["target_logit_drop_aoc_normalized"]),
                    _format_stats(stats["road_morf_mean_drop"]),
                    _format_stats(stats["road_morf_mean_consistency"]),
                    _format_stats(stats["rank"]),
                    _format_stats(stats["runtime_s"]),
                    _format_stats(stats["evaluation_runtime_s"]),
                    _format_stats(stats["abs_error"]),
                ]
            )
            + " |"
        )

    core_summary = summary["core_summary"]
    lines.extend(
        [
            "",
            "## Core Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| clean_top1_logit | {_format_stats(core_summary['clean_top1_logit'])} |",
            f"| clean_top1_prob | {_format_stats(core_summary['clean_top1_prob'])} |",
            f"| n_pixels_total | {_format_stats(core_summary['n_pixels_total'])} |",
            f"| n_percentiles | {_format_stats(core_summary['n_percentiles'])} |",
            f"| core_runtime_s | {_format_stats(core_summary['core_runtime_s'])} |",
        ]
    )

    if figures:
        lines.extend(["", "## Figures", ""])
        for key in (
            "road_summary",
            "road_distributions",
            "road_curves",
            "road_family_envelope",
            "road_cheap_ig_heatmaps",
            "road_cheap_ig_delta_boxplots",
            "road_pairwise_wins",
        ):
            path = figures.get(key)
            if path:
                lines.extend([f"### {key}", "", f"![]({_relative_markdown_path(path)})", ""])

    lines.extend(
        [
            "## Per-Image Scores",
            "",
            _build_per_image_markdown_table(rows, method_specs),
            "",
            "## Pairwise Win Rate",
            "",
            _build_pairwise_markdown_table(summary["pairwise_win_rate"], method_specs),
            "",
        ]
    )

    if preview_sections:
        lines.extend(
            [
                "## Visual Preview",
                "",
                "Fixed 5 images shared across all methods; rows are identical across tables.",
                "",
            ]
        )
        for section in preview_sections:
            lines.extend([f"### {section['title']}", "", section["markdown_table"], ""])

    return "\n".join(lines).strip()


def _plot_road_summary(summary, method_specs):
    labels = [spec["name"] for spec in method_specs]
    means = [summary["method_summaries"][label]["target_logit_drop_aoc"]["mean"] for label in labels]
    stds = [summary["method_summaries"][label]["target_logit_drop_aoc"]["std"] for label in labels]
    order = np.argsort(np.asarray(means, dtype=np.float64))[::-1]
    sorted_labels = [labels[idx] for idx in order.tolist()]
    sorted_means = [means[idx] for idx in order.tolist()]
    sorted_stds = [stds[idx] for idx in order.tolist()]
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.42 * len(sorted_labels) + 1.5)), constrained_layout=True)
    y = np.arange(len(sorted_labels))
    colors = [_method_color(method_specs[labels.index(label)]) for label in sorted_labels]
    ax.hlines(y, np.asarray(sorted_means) - np.asarray(sorted_stds), np.asarray(sorted_means) + np.asarray(sorted_stds), color=colors, linewidth=2.0, alpha=0.9)
    ax.scatter(sorted_means, y, color=colors, s=70, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Target logit drop AOC (higher is better)")
    ax.set_title("ROAD MoRF target-logit summary by method")
    ax.grid(axis="x", alpha=0.3)
    return fig


def _plot_road_distributions(rows, method_specs):
    labels = [spec["name"] for spec in method_specs]
    data = []
    for spec in method_specs:
        method_rows = [row for row in rows if row["method_name"] == spec["name"]]
        values = [row["target_logit_drop_aoc"] for row in method_rows if row["target_logit_drop_aoc"] == row["target_logit_drop_aoc"]]
        data.append(values)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * len(labels)), 5.5), constrained_layout=True)
    box = ax.boxplot(data, patch_artist=True, labels=labels)
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, len(labels)))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Target logit drop AOC")
    ax.set_title("ROAD MoRF target-logit AOC distribution by method")
    ax.grid(axis="y", alpha=0.3)
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")
    return fig


def _plot_road_curves(rows, percentiles, method_specs):
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 9.0), constrained_layout=True, sharex=True)
    x = np.asarray(percentiles, dtype=np.int64)
    for color_idx, spec in enumerate(method_specs):
        method_rows = [row for row in rows if row["method_name"] == spec["name"]]
        logit_drop_curves = np.asarray([row["target_logit_drop_curve"] for row in method_rows], dtype=np.float64)
        consistency_curves = np.asarray([row["consistency_curve"] for row in method_rows], dtype=np.float64)
        if logit_drop_curves.size == 0 or consistency_curves.size == 0:
            continue
        color = plt.cm.tab20(color_idx % 20)
        axes[0].plot(
            x,
            np.nanmean(logit_drop_curves, axis=0),
            marker="o",
            label=spec["name"],
            color=color,
        )
        axes[1].plot(
            x,
            np.nanmean(consistency_curves, axis=0),
            marker="o",
            label=spec["name"],
            color=color,
        )
    axes[0].set_ylabel("Target logit drop")
    axes[0].set_title("ROAD MoRF target-logit drop curves")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("Removed pixels (%)")
    axes[1].set_ylabel("Top-1 consistency")
    axes[1].set_title("ROAD MoRF consistency curves")
    axes[1].set_xticks(x)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    return fig


def _plot_road_family_envelope(rows, percentiles, method_specs):
    cheap_specs = _cheap_specs_sorted(method_specs)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 9.0), constrained_layout=True, sharex=True)
    x = np.asarray(percentiles, dtype=np.int64)

    baseline_names = [spec["name"] for spec in method_specs if spec["kind"] in {"ig", "naa"}]
    for baseline_name in baseline_names:
        spec = next(spec for spec in method_specs if spec["name"] == baseline_name)
        baseline_rows = [row for row in rows if row["method_name"] == baseline_name]
        logit_curves = np.asarray([row["target_logit_drop_curve"] for row in baseline_rows], dtype=np.float64)
        consistency_curves = np.asarray([row["consistency_curve"] for row in baseline_rows], dtype=np.float64)
        if logit_curves.size > 0:
            axes[0].plot(x, np.nanmean(logit_curves, axis=0), marker="o", linewidth=2.4, label=baseline_name, color=_method_color(spec))
        if consistency_curves.size > 0:
            axes[1].plot(x, np.nanmean(consistency_curves, axis=0), marker="o", linewidth=2.4, label=baseline_name, color=_method_color(spec))

    cheap_logit_means = []
    cheap_consistency_means = []
    for spec in cheap_specs:
        method_rows = [row for row in rows if row["method_name"] == spec["name"]]
        logit_curves = np.asarray([row["target_logit_drop_curve"] for row in method_rows], dtype=np.float64)
        consistency_curves = np.asarray([row["consistency_curve"] for row in method_rows], dtype=np.float64)
        if logit_curves.size == 0 or consistency_curves.size == 0:
            continue
        cheap_logit_means.append(np.nanmean(logit_curves, axis=0))
        cheap_consistency_means.append(np.nanmean(consistency_curves, axis=0))

    if cheap_logit_means:
        cheap_logit_means = np.asarray(cheap_logit_means, dtype=np.float64)
        cheap_consistency_means = np.asarray(cheap_consistency_means, dtype=np.float64)
        axes[0].fill_between(x, np.nanmin(cheap_logit_means, axis=0), np.nanmax(cheap_logit_means, axis=0), color="tab:red", alpha=0.15, label="Cheap-IG family range")
        axes[0].plot(x, np.nanmean(cheap_logit_means, axis=0), color="tab:red", linewidth=2.8, marker="o", label="Cheap-IG family mean")
        axes[1].fill_between(x, np.nanmin(cheap_consistency_means, axis=0), np.nanmax(cheap_consistency_means, axis=0), color="tab:red", alpha=0.15, label="Cheap-IG family range")
        axes[1].plot(x, np.nanmean(cheap_consistency_means, axis=0), color="tab:red", linewidth=2.8, marker="o", label="Cheap-IG family mean")

    for fill_key in _cheap_fill_order(method_specs):
        fill_specs = [spec for spec in cheap_specs if _cheap_fill_key(spec) == fill_key]
        fill_color = _cheap_fill_color(fill_key)
        label = _cheap_fill_label(fill_key)
        fill_logit_means = []
        fill_consistency_means = []
        for spec in fill_specs:
            method_rows = [row for row in rows if row["method_name"] == spec["name"]]
            logit_curves = np.asarray([row["target_logit_drop_curve"] for row in method_rows], dtype=np.float64)
            consistency_curves = np.asarray([row["consistency_curve"] for row in method_rows], dtype=np.float64)
            if logit_curves.size == 0 or consistency_curves.size == 0:
                continue
            fill_logit_means.append(np.nanmean(logit_curves, axis=0))
            fill_consistency_means.append(np.nanmean(consistency_curves, axis=0))
        if fill_logit_means:
            axes[0].plot(x, np.nanmean(np.asarray(fill_logit_means, dtype=np.float64), axis=0), linestyle="--", linewidth=1.8, color=fill_color, alpha=0.9, label=label)
            axes[1].plot(x, np.nanmean(np.asarray(fill_consistency_means, dtype=np.float64), axis=0), linestyle="--", linewidth=1.8, color=fill_color, alpha=0.9, label=label)

    axes[0].set_ylabel("Target logit drop")
    axes[0].set_title("Cheap-IG family envelope vs baselines")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2)

    axes[1].set_xlabel("Removed pixels (%)")
    axes[1].set_ylabel("Top-1 consistency")
    axes[1].set_title("Cheap-IG family consistency envelope vs baselines")
    axes[1].set_xticks(x)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)
    return fig


def _plot_cheap_ig_heatmaps(summary, method_specs):
    cheap_specs = _cheap_specs_sorted(method_specs)
    if not cheap_specs:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_axis_off()
        ax.set_title("No cheap-IG methods")
        return fig

    row_keys = _cheap_fill_order(method_specs)
    col_keys = _cheap_k_order(method_specs)
    ig_mean = summary["method_summaries"].get("IG", {}).get("target_logit_drop_aoc", {}).get("mean", float("nan"))
    naa_mean = summary["method_summaries"].get("NAA", {}).get("target_logit_drop_aoc", {}).get("mean", float("nan"))

    mean_matrix = np.full((len(row_keys), len(col_keys)), np.nan, dtype=np.float64)
    delta_ig_matrix = np.full_like(mean_matrix, np.nan)
    delta_naa_matrix = np.full_like(mean_matrix, np.nan)
    win_baseline_matrix = np.full_like(mean_matrix, np.nan)

    for spec in cheap_specs:
        row_idx = row_keys.index(_cheap_fill_key(spec))
        col_idx = col_keys.index(int(spec.get("selection_top_k", -1)))
        method_name = spec["name"]
        method_summary = summary["method_summaries"].get(method_name, {})
        mean_value = method_summary.get("target_logit_drop_aoc", {}).get("mean", float("nan"))
        mean_matrix[row_idx, col_idx] = mean_value
        delta_ig_matrix[row_idx, col_idx] = mean_value - float(ig_mean)
        delta_naa_matrix[row_idx, col_idx] = mean_value - float(naa_mean)
        pairwise = summary["pairwise_win_rate"].get(method_name, {})
        wins = []
        for baseline_name in ("IG", "NAA"):
            value = pairwise.get(baseline_name, float("nan"))
            if value == value:
                wins.append(float(value))
        if wins:
            win_baseline_matrix[row_idx, col_idx] = float(np.mean(wins))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), constrained_layout=True)
    panels = (
        (mean_matrix, "Cheap-IG mean AOC", "magma"),
        (delta_ig_matrix, "AOC advantage vs IG", "coolwarm"),
        (delta_naa_matrix, "AOC advantage vs NAA", "coolwarm"),
        (win_baseline_matrix, "Mean win-rate vs IG/NAA", "viridis"),
    )
    row_labels = [_cheap_fill_label(key) for key in row_keys]
    col_labels = [f"k={int(value)}" for value in col_keys]
    for ax, (matrix, title, cmap) in zip(axes.reshape(-1), panels):
        if "advantage" in title:
            vmax = np.nanmax(np.abs(matrix)) if np.isfinite(matrix).any() else 1.0
            vmin = -vmax
        elif "win-rate" in title:
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = None, None
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_title(title)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                label = "n/a" if value != value else f"{value:.3f}"
                ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _plot_cheap_ig_delta_boxplots(rows, method_specs):
    cheap_specs = _cheap_specs_sorted(method_specs)
    if not cheap_specs:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_axis_off()
        ax.set_title("No cheap-IG methods")
        return fig

    by_image_method = {}
    for row in rows:
        by_image_method.setdefault(row["image_path"], {})[row["method_name"]] = row

    labels = [_cheap_variant_short_label(spec) for spec in cheap_specs]
    colors = [_cheap_fill_color(_cheap_fill_key(spec)) for spec in cheap_specs]
    delta_vs_ig = []
    delta_vs_naa = []
    for spec in cheap_specs:
        method_name = spec["name"]
        ig_values = []
        naa_values = []
        for image_rows in by_image_method.values():
            candidate = image_rows.get(method_name)
            ig_row = image_rows.get("IG")
            naa_row = image_rows.get("NAA")
            if candidate is not None and ig_row is not None:
                left = candidate.get("target_logit_drop_aoc")
                right = ig_row.get("target_logit_drop_aoc")
                if left == left and right == right:
                    ig_values.append(float(left) - float(right))
            if candidate is not None and naa_row is not None:
                left = candidate.get("target_logit_drop_aoc")
                right = naa_row.get("target_logit_drop_aoc")
                if left == left and right == right:
                    naa_values.append(float(left) - float(right))
        delta_vs_ig.append(ig_values)
        delta_vs_naa.append(naa_values)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(5.0, 0.5 * len(labels) + 1.5)), constrained_layout=True, sharey=True)
    for ax, series, title in (
        (axes[0], delta_vs_ig, "Per-image AOC advantage vs IG"),
        (axes[1], delta_vs_naa, "Per-image AOC advantage vs NAA"),
    ):
        safe_series = [values if values else [float("nan")] for values in series]
        box = ax.boxplot(safe_series, vert=False, patch_artist=True, labels=labels, showmeans=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for median in box["medians"]:
            median.set_color("black")
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.set_title(title)
        ax.set_xlabel("Target logit drop AOC delta")
        ax.grid(axis="x", alpha=0.3)
    axes[0].invert_yaxis()
    return fig


def _plot_pairwise_win_heatmap(summary, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    matrix = np.full((len(method_names), len(method_names)), np.nan, dtype=np.float64)
    pairwise = summary["pairwise_win_rate"]
    for row_idx, left in enumerate(method_names):
        for col_idx, right in enumerate(method_names):
            matrix[row_idx, col_idx] = pairwise[left].get(right, float("nan"))

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=25, ha="right")
    ax.set_yticklabels(method_names)
    ax.set_title("Pairwise win rate by image")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "—" if row_idx == col_idx else ("n/a" if value != value else f"{value:.2f}")
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Win rate")
    return fig


def _pairwise_win_rate(rows, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_path"], {})[row["method_name"]] = row["score"]

    result = {left: {} for left in method_names}
    for left in method_names:
        for right in method_names:
            if left == right:
                result[left][right] = float("nan")
                continue
            wins = 0
            total = 0
            for image_scores in by_image.values():
                left_score = image_scores.get(left)
                right_score = image_scores.get(right)
                if left_score is None or right_score is None:
                    continue
                if left_score != left_score or right_score != right_score:
                    continue
                total += 1
                if left_score > right_score:
                    wins += 1
            result[left][right] = float(wins / total) if total > 0 else float("nan")
    return result


def _build_per_image_markdown_table(rows, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_name"], {})[row["method_name"]] = row.get("target_logit_drop_aoc")
    lines = [
        "| Image | " + " | ".join(method_names) + " |",
        "| --- | " + " | ".join("---:" for _ in method_names) + " |",
    ]
    for image_name in sorted(by_image):
        scores = by_image[image_name]
        values = [_format_number(scores.get(name)) for name in method_names]
        lines.append(f"| {image_name} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _build_pairwise_markdown_table(pairwise_win_rate, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    lines = [
        "| Method | " + " | ".join(method_names) + " |",
        "| --- | " + " | ".join("---:" for _ in method_names) + " |",
    ]
    for left in method_names:
        row = [_format_number(pairwise_win_rate[left].get(right)) if left != right else "—" for right in method_names]
        lines.append(f"| {left} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def _method_color(spec):
    if spec["kind"] == "ig":
        return "tab:blue"
    if spec["kind"] == "naa":
        return "tab:gray"
    return _cheap_fill_color(_cheap_fill_key(spec))


def _cheap_specs_sorted(method_specs):
    cheap_specs = [spec for spec in method_specs if spec["kind"] == "cheap_ig"]
    return sorted(
        cheap_specs,
        key=lambda spec: (
            _cheap_fill_order(method_specs).index(_cheap_fill_key(spec)),
            int(spec.get("selection_top_k", 0)),
            spec["name"],
        ),
    )


def _cheap_fill_order(method_specs):
    present = []
    for spec in method_specs:
        if spec["kind"] != "cheap_ig":
            continue
        key = _cheap_fill_key(spec)
        if key not in present:
            present.append(key)
    preferred = ["zero", "rho0.8", "rho1.0"]
    ordered = [key for key in preferred if key in present]
    ordered.extend(key for key in present if key not in ordered)
    return ordered


def _cheap_k_order(method_specs):
    return sorted({int(spec.get("selection_top_k", 0)) for spec in method_specs if spec["kind"] == "cheap_ig"})


def _cheap_fill_key(spec):
    fill_mode = str(spec.get("fill_mode", "zero"))
    if fill_mode == "zero":
        return "zero"
    rho = float(spec.get("fill_rho", 0.8))
    if abs(rho - 0.8) <= 1e-9:
        return "rho0.8"
    if abs(rho - 1.0) <= 1e-9:
        return "rho1.0"
    return f"{fill_mode}/rho{rho:g}"


def _cheap_fill_label(fill_key):
    mapping = {
        "zero": "no tail",
        "rho0.8": "NAA tail, rho=0.8",
        "rho1.0": "NAA tail, rho=1.0",
    }
    return mapping.get(fill_key, fill_key)


def _cheap_fill_color(fill_key):
    mapping = {
        "zero": "tab:green",
        "rho0.8": "tab:red",
        "rho1.0": "tab:purple",
    }
    return mapping.get(fill_key, "tab:brown")


def _cheap_variant_short_label(spec):
    return f"k={int(spec.get('selection_top_k', 0))}, {_cheap_fill_label(_cheap_fill_key(spec))}"


def _build_preview_markdown_for_columns(rows, columns):
    header = " | ".join(column["title"] for column in columns)
    separator = " | ".join("---" for _ in columns)
    lines = [
        f"| {header} |",
        f"| {separator} |",
    ]
    for row in rows:
        refs = []
        for column in columns:
            output_path = row.get(f"{column['key']}_output_path")
            if not output_path:
                refs = []
                break
            refs.append(f"![]({_relative_markdown_path(output_path)})")
        if refs:
            lines.append(f"| {' | '.join(refs)} |")
    return "\n".join(lines)


def _build_raw_image_figure(image_np, title):
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(image_np, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    return fig


def _build_spatial_overlay_figure(*, image_np, spatial_map, title):
    signed_map = np.asarray(spatial_map, dtype=np.float32)
    signed_map = _normalize_map(signed_map)
    if tuple(signed_map.shape) != tuple(image_np.shape[:2]):
        signed_map = IG._resize_map_nearest(signed_map, image_np.shape[:2]).astype(np.float32, copy=False)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.imshow(image_np, interpolation="nearest")
    heat = ax.imshow(signed_map, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title)
    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized attribution", rotation=90)
    return fig


def _normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs > 0.0:
        arr = arr / max_abs
    return arr


def _percentile_pixel_counts(*, height, width, percentiles):
    n_pixels = int(height) * int(width)
    counts = []
    for percentile in percentiles:
        counts.append(max(1, min(n_pixels, int(np.ceil(float(percentile) / 100.0 * float(n_pixels))))))
    return counts


def _stable_descending_order(scores):
    values = np.asarray(scores, dtype=np.float64).copy()
    idx = np.arange(values.size, dtype=np.int64)
    values[~np.isfinite(values)] = -np.inf
    order = np.lexsort((idx, -values))
    return order.astype(np.int64, copy=False)


def _normalize_percentiles(percentiles):
    if percentiles is None:
        raise ValueError("percentiles must not be None.")
    values = [int(value) for value in percentiles]
    if not values:
        raise ValueError("percentiles must contain at least one value.")
    for value in values:
        if value < 1 or value > 99:
            raise ValueError(f"Each percentile must be in [1, 99], got {value}.")
    return tuple(values)


def _normalize_noise(noise):
    value = float(noise)
    if value < 0.0:
        raise ValueError(f"noise must be >= 0, got {value}.")
    return value


def _normalize_method_specs(method_specs):
    normalized = []
    seen_ids = set()
    for raw_spec in method_specs:
        spec = canonicalize_baseline_config(raw_spec)
        spec["kind"] = str(spec["kind"])
        spec["name"] = str(spec.get("name") or _default_method_name(spec))
        spec["id"] = _method_id(spec)
        if spec["id"] in seen_ids:
            raise ValueError(f"Duplicate method spec id: {spec['id']}. Please use unique names/params.")
        seen_ids.add(spec["id"])
        normalized.append(spec)
    return normalized


def _default_method_name(spec):
    kind = str(spec["kind"])
    baseline_fragment = baseline_method_fragment(
        spec.get("baseline_mode", "zero"),
        baseline_rgb=spec.get("baseline_rgb"),
        blur_sigma=spec.get("baseline_blur_sigma", DEFAULT_BLUR_SIGMA),
    )
    if kind != "cheap_ig":
        base_name = kind.upper() if kind in {"ig", "naa"} else kind
        return f"{base_name}{baseline_fragment}"
    seg_start = spec.get("segment_start", 0.0)
    seg_end = spec.get("segment_end", 0.1)
    selection_mode = spec.get("selection_mode", "signed")
    selection_top_k = spec.get("selection_top_k", 5000)
    fill_mode = spec.get("fill_mode", "zero")
    if str(fill_mode) == "zero":
        fill_fragment = ""
    else:
        fill_fragment = f"/fill-{fill_mode}-rho{float(spec.get('fill_rho', 0.8)):g}"
    return (
        f"cheap-ig[{seg_start:g},{seg_end:g}]/{selection_mode}/k{int(selection_top_k)}"
        f"{fill_fragment}{baseline_fragment}"
    )


def _method_id(spec):
    material = _stable_spec_string(spec)
    digest = hashlib.md5(material.encode("utf-8")).hexdigest()[:10]
    return f"{_safe_slug(spec['name'])}_{digest}"


def _stable_spec_string(value):
    if isinstance(value, dict):
        items = ",".join(f"{key}:{_stable_spec_string(value[key])}" for key in sorted(value))
        return "{" + items + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_stable_spec_string(item) for item in value) + "]"
    return repr(value)


def _clear_all_backend_caches():
    for module in (IG, NAA):
        clear_fn = getattr(module, "clear_backend_cache", None)
        if callable(clear_fn):
            clear_fn()


def _stats_record(values):
    numeric = np.asarray([float(v) for v in values if v is not None and float(v) == float(v)], dtype=np.float64)
    if numeric.size == 0:
        return {
            "n_ok": 0,
            "n_total": int(len(values)),
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "n_ok": int(numeric.size),
        "n_total": int(len(values)),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()),
        "median": float(np.median(numeric)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
    }


def _format_number(value, digits=4):
    if value is None:
        return "n/a"
    value = float(value)
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def _format_stats(stats):
    if stats["n_ok"] == 0:
        return "n/a"
    return f"{stats['mean']:.4f} +- {stats['std']:.4f}"


def _safe_slug(text, max_len=64):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = "record"
    return cleaned[:max_len]


def _prepare_output_dir(output_dir, run_name):
    base_dir = Path(output_dir)
    if not base_dir.is_absolute():
        base_dir = Path.cwd() / base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    candidate = base_dir / run_name
    suffix = 2
    while candidate.exists():
        candidate = base_dir / f"{run_name}_{suffix}"
        suffix += 1

    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _prepare_existing_output_dir(output_dir):
    path = Path(output_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _relative_markdown_path(path):
    path = Path(path)
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _pretty_json(payload):
    return json.dumps(_normalize_for_hash(payload), indent=2, ensure_ascii=True, sort_keys=True)


def _normalize_for_hash(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_hash(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _config_hash(config):
    payload = json.dumps(_normalize_for_hash(config), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]


def _array_sidecar_path(*, cache_root, namespace, label, sidecar_key, kind):
    root = Path(cache_root)
    filename = f"{_safe_slug(label, max_len=48)}_{sidecar_key}_{_safe_slug(kind, max_len=24)}.npz"
    path = root / namespace / "arrays" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _save_array_sidecar(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, array=np.asarray(array))


def _load_array_sidecar(path):
    with np.load(Path(path), allow_pickle=False) as data:
        return np.asarray(data["array"])


def _derived_seed(base_seed, *parts):
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts])
    return int(hashlib.md5(payload.encode("utf-8")).hexdigest()[:8], 16)
