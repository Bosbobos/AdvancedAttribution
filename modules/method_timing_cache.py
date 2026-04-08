from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np


SCHEMA_VERSION = 1


def image_label(image_path):
    return Path(image_path).stem


def image_signature(image_path):
    path = Path(image_path)
    if path.exists():
        stat = path.stat()
        return {
            "path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return {
        "path": str(path),
        "size": None,
        "mtime_ns": None,
    }


def current_device_label(device):
    return str(device) if device is not None else "n/a"


def _normalize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    return value


def _stable_json(value):
    return json.dumps(_normalize(value), sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _config_hash(config):
    return hashlib.md5(_stable_json(config).encode("utf-8")).hexdigest()[:12]


def _safe_slug(text, max_len=48):
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = "record"
    return cleaned[:max_len]


def _cache_path(cache_root, namespace, config, label="record"):
    root = Path(cache_root)
    payload_hash = _config_hash(config)
    filename = f"{_safe_slug(label)}_{payload_hash}.json"
    return root / namespace / filename


def _load_json(path):
    return json.loads(Path(path).read_text())


def _save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize(payload), indent=2, ensure_ascii=True, sort_keys=True))


def _timing_result_from_record(record, cache_path, repeats, result=None, from_cache=True):
    runs = [float(value) for value in record.get("runs", [])][:repeats]
    error = record.get("error")
    return {
        "runs": runs,
        "mean_s": mean(runs) if runs and error is None else None,
        "error": error,
        "result": result if result is not None else record.get("result"),
        "cache_path": str(cache_path),
        "from_cache": from_cache,
    }


def time_call_cached(
    *,
    cache_root,
    namespace,
    config,
    label,
    fn,
    repeats=1,
    refresh=False,
    clear_before=None,
    required_device=None,
    current_device=None,
    result_serializer=None,
):
    cache_path = _cache_path(cache_root, namespace, config, label=label)
    normalized_config = _normalize(config)
    record = None

    if cache_path.exists() and not refresh:
        record = _load_json(cache_path)
        if record.get("schema_version") == SCHEMA_VERSION and record.get("config") == normalized_config:
            if record.get("error") is not None:
                return _timing_result_from_record(record, cache_path, repeats, from_cache=True)
            if len(record.get("runs", [])) >= repeats:
                return _timing_result_from_record(record, cache_path, repeats, from_cache=True)
        else:
            record = None

    existing_runs = []
    if record and record.get("error") is None:
        existing_runs = [float(value) for value in record.get("runs", [])]

    if required_device and current_device != required_device:
        raise RuntimeError(
            f"Current device is `{current_device}`, but cache miss requires `{required_device}` for {namespace}."
        )

    timings = list(existing_runs)
    error = None
    last_result = record.get("result") if record else None

    while len(timings) < repeats:
        if clear_before is not None:
            clear_before()
        start = perf_counter()
        try:
            last_result = fn()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            break
        timings.append(perf_counter() - start)

    stored_result = result_serializer(last_result) if result_serializer and last_result is not None else last_result
    new_record = {
        "schema_version": SCHEMA_VERSION,
        "namespace": namespace,
        "config": normalized_config,
        "runs": timings,
        "error": error,
        "result": stored_result,
        "required_device": required_device,
        "actual_device": current_device,
    }
    _save_json(cache_path, new_record)
    return _timing_result_from_record(new_record, cache_path, repeats, result=last_result, from_cache=False)


def load_or_compute_cached_value(
    *,
    cache_root,
    namespace,
    config,
    label,
    compute_fn,
    refresh=False,
    required_device=None,
    current_device=None,
):
    cache_path = _cache_path(cache_root, namespace, config, label=label)
    normalized_config = _normalize(config)

    if cache_path.exists() and not refresh:
        record = _load_json(cache_path)
        if record.get("schema_version") == SCHEMA_VERSION and record.get("config") == normalized_config:
            return {
                "value": record.get("value"),
                "duration_s": record.get("duration_s"),
                "error": record.get("error"),
                "cache_path": str(cache_path),
                "from_cache": True,
            }

    if required_device and current_device != required_device:
        raise RuntimeError(
            f"Current device is `{current_device}`, but cache miss requires `{required_device}` for {namespace}."
        )

    start = perf_counter()
    value = None
    error = None
    try:
        value = compute_fn()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    duration_s = perf_counter() - start

    record = {
        "schema_version": SCHEMA_VERSION,
        "namespace": namespace,
        "config": normalized_config,
        "value": _normalize(value),
        "duration_s": duration_s,
        "error": error,
        "required_device": required_device,
        "actual_device": current_device,
    }
    _save_json(cache_path, record)
    return {
        "value": value,
        "duration_s": duration_s,
        "error": error,
        "cache_path": str(cache_path),
        "from_cache": False,
    }


def benchmark_classifiers_cached(
    *,
    image_paths,
    layer_name,
    n_steps,
    repeats,
    clear_every,
    ig_module,
    naa_module,
    clear_all_caches_fn,
    cache_root,
    target_device=None,
    refresh=False,
):
    ig_device = current_device_label(getattr(ig_module, "DEVICE", None))
    naa_device = current_device_label(getattr(naa_module, "DEVICE", None))
    rows = []

    for image_path in image_paths:
        image_key = image_signature(image_path)
        common_config = {
            "method_family": "classifier",
            "image": image_key,
            "layer_name": layer_name,
            "n_steps": int(n_steps),
            "clear_every": int(clear_every),
            "target_device": target_device,
        }

        ig_timing = time_call_cached(
            cache_root=cache_root,
            namespace="classifier_ig",
            config={**common_config, "method": "IG"},
            label=f"{image_label(image_path)}_ig",
            fn=lambda image_path=image_path: ig_module.run_conductance_pipeline(
                image_path=image_path,
                layer_name=layer_name,
                n_steps=n_steps,
                top_n=0,
                clear_every=clear_every,
                verbose=False,
                show_total_plot=False,
                show_filter_plots=False,
            ),
            repeats=repeats,
            refresh=refresh,
            clear_before=clear_all_caches_fn,
            required_device=target_device,
            current_device=ig_device,
            result_serializer=lambda _result: None,
        )
        naa_timing = time_call_cached(
            cache_root=cache_root,
            namespace="classifier_naa",
            config={**common_config, "method": "NAA"},
            label=f"{image_label(image_path)}_naa",
            fn=lambda image_path=image_path: naa_module.run_attribution_pipeline(
                image_path=image_path,
                layer_name=layer_name,
                n_steps=n_steps,
                top_n=0,
                clear_every=clear_every,
                verbose=False,
                show_total_plot=False,
                show_filter_plots=False,
            ),
            repeats=repeats,
            refresh=refresh,
            clear_before=clear_all_caches_fn,
            required_device=target_device,
            current_device=naa_device,
            result_serializer=lambda _result: None,
        )
        rows.append(
            {
                "image_path": image_path,
                "image_label": image_label(image_path),
                "ig_s": ig_timing["mean_s"],
                "naa_s": naa_timing["mean_s"],
                "ig_runs": ig_timing["runs"],
                "naa_runs": naa_timing["runs"],
                "ig_error": ig_timing["error"],
                "naa_error": naa_timing["error"],
                "ig_cache_path": ig_timing["cache_path"],
                "naa_cache_path": naa_timing["cache_path"],
                "ig_from_cache": ig_timing["from_cache"],
                "naa_from_cache": naa_timing["from_cache"],
            }
        )

    return {
        "kind": "classifier",
        "layer_name": layer_name,
        "n_steps": n_steps,
        "repeats": repeats,
        "rows": rows,
        "target_device": target_device,
        "cache_root": str(cache_root),
        "actual_devices": {"IG": ig_device, "NAA": naa_device},
    }


def benchmark_detectors_cached(
    *,
    image_paths,
    layer_name,
    mode,
    roi_top_k,
    n_steps,
    repeats,
    clear_every,
    ig_det_module,
    naa_det_module,
    clear_all_caches_fn,
    cache_root,
    target_device=None,
    refresh=False,
):
    ig_device = current_device_label(getattr(ig_det_module, "DEVICE", None))
    naa_device = current_device_label(getattr(naa_det_module, "DEVICE", None))
    rows = []

    for image_path in image_paths:
        image_key = image_signature(image_path)
        common_config = {
            "method_family": "detector",
            "image": image_key,
            "layer_name": layer_name,
            "mode": mode,
            "roi_top_k": int(roi_top_k),
            "n_steps": int(n_steps),
            "clear_every": int(clear_every),
            "target_device": target_device,
        }

        ig_timing = time_call_cached(
            cache_root=cache_root,
            namespace="detector_ig",
            config={**common_config, "method": "IG"},
            label=f"{image_label(image_path)}_ig_det",
            fn=lambda image_path=image_path: ig_det_module.run_detector_conductance(
                image_path=image_path,
                mode=mode,
                layer_name=layer_name,
                n_steps=n_steps,
                top_n=0,
                roi_top_k=roi_top_k,
                clear_every=clear_every,
                verbose=False,
                show_total_plot=False,
                show_filter_plots=False,
                show_target_box=False,
            ),
            repeats=repeats,
            refresh=refresh,
            clear_before=clear_all_caches_fn,
            required_device=target_device,
            current_device=ig_device,
            result_serializer=lambda _result: None,
        )
        naa_timing = time_call_cached(
            cache_root=cache_root,
            namespace="detector_naa",
            config={**common_config, "method": "NAA"},
            label=f"{image_label(image_path)}_naa_det",
            fn=lambda image_path=image_path: naa_det_module.run_attribution_pipeline(
                image_path=image_path,
                mode=mode,
                layer_name=layer_name,
                n_steps=n_steps,
                top_n=0,
                roi_top_k=roi_top_k,
                clear_every=clear_every,
                verbose=False,
                show_total_plot=False,
                show_filter_plots=False,
                show_target_box=False,
            ),
            repeats=repeats,
            refresh=refresh,
            clear_before=clear_all_caches_fn,
            required_device=target_device,
            current_device=naa_device,
            result_serializer=lambda _result: None,
        )
        rows.append(
            {
                "image_path": image_path,
                "image_label": image_label(image_path),
                "ig_s": ig_timing["mean_s"],
                "naa_s": naa_timing["mean_s"],
                "ig_runs": ig_timing["runs"],
                "naa_runs": naa_timing["runs"],
                "ig_error": ig_timing["error"],
                "naa_error": naa_timing["error"],
                "ig_cache_path": ig_timing["cache_path"],
                "naa_cache_path": naa_timing["cache_path"],
                "ig_from_cache": ig_timing["from_cache"],
                "naa_from_cache": naa_timing["from_cache"],
            }
        )

    return {
        "kind": "detector",
        "layer_name": layer_name,
        "mode": mode,
        "roi_top_k": roi_top_k,
        "n_steps": n_steps,
        "repeats": repeats,
        "rows": rows,
        "target_device": target_device,
        "cache_root": str(cache_root),
        "actual_devices": {"IG": ig_device, "NAA": naa_device},
    }


def _fit_custom_scale(sums, targets):
    sums = np.asarray(sums, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    return float(np.dot(sums, targets) / (np.dot(sums, sums) + 1e-12))


def _selection_label(selection_mode, selection_top_k):
    return f"{selection_mode}_{int(selection_top_k)}"


def covariance_sparse_methods_from_cache(
    *,
    cov_ns,
    cache,
    selection_mode="signed",
    selection_top_k=5000,
    exact_segment_scale=None,
    sparse_scale=None,
):
    base_methods = cov_ns["compute_method_vectors"](cache, top_n=0)
    approx_naa = base_methods["approx_naa"].astype(np.float64, copy=True)
    exact_segment = base_methods["exact_segment"].astype(np.float64)

    sparse = np.zeros_like(exact_segment)
    if selection_top_k > 0 and approx_naa.size > 0:
        selection_top_k = int(min(selection_top_k, approx_naa.size))
        if selection_mode == "signed":
            pos_idx = np.flatnonzero(approx_naa > 0)
            neg_idx = np.flatnonzero(approx_naa < 0)
            if pos_idx.size > 0:
                k_pos = min(selection_top_k, pos_idx.size)
                chosen_pos = pos_idx[np.argpartition(approx_naa[pos_idx], -k_pos)[-k_pos:]]
                sparse[chosen_pos] = exact_segment[chosen_pos]
            if neg_idx.size > 0:
                k_neg = min(selection_top_k, neg_idx.size)
                chosen_neg = neg_idx[np.argpartition(np.abs(approx_naa[neg_idx]), -k_neg)[-k_neg:]]
                sparse[chosen_neg] = exact_segment[chosen_neg]
        elif selection_mode == "unsigned":
            idx = np.argpartition(np.abs(approx_naa), -selection_top_k)[-selection_top_k:]
            sparse[idx] = exact_segment[idx]
        else:
            raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    exact_segment_scaled = exact_segment if exact_segment_scale is None else exact_segment * float(exact_segment_scale)
    sparse_scaled = sparse if sparse_scale is None else sparse * float(sparse_scale)
    return {
        "approx_naa": approx_naa,
        "exact_segment_scaled": exact_segment_scaled,
        "topk_exact_only_segment_scaled": sparse_scaled,
    }


def fit_covariance_sparse_scales_from_cache_paths(
    *,
    cov_ns,
    cache_paths,
    selection_mode="signed",
    selection_top_k=5000,
):
    exact_segment_sums = []
    sparse_sums = []
    targets = []

    for cache_path in cache_paths:
        cache = cov_ns["load_cache"](cache_path)
        method_vectors = covariance_sparse_methods_from_cache(
            cov_ns=cov_ns,
            cache=cache,
            selection_mode=selection_mode,
            selection_top_k=selection_top_k,
        )
        fx_delta = float(cache["fx_delta"][0])
        exact_segment_sums.append(float(method_vectors["exact_segment_scaled"].sum()))
        sparse_sums.append(float(method_vectors["topk_exact_only_segment_scaled"].sum()))
        targets.append(fx_delta)

    return {
        "exact_segment_scale": _fit_custom_scale(exact_segment_sums, targets),
        "sparse_scale": _fit_custom_scale(sparse_sums, targets),
    }


def _covariance_cache_location(cov_ns, image_path, layer_name, n_steps, segment_start, segment_end):
    image_path = Path(image_path)
    cache_dir = cov_ns["_cache_dir"](layer_name, n_steps, segment_start, segment_end)
    cache_path = cov_ns["_cache_path"](cache_dir, image_path)
    return cache_dir, cache_path


def _run_covariance_prepare_once(
    *,
    cov_ns,
    image_path,
    layer_name,
    n_steps,
    segment_start,
    segment_end,
    clear_every,
):
    image_path = Path(image_path)
    cache_dir, cache_path = _covariance_cache_location(
        cov_ns,
        image_path,
        layer_name,
        n_steps,
        segment_start,
        segment_end,
    )
    payload = cov_ns["compute_cache_for_image"](
        model=cov_ns["model"],
        class_names=cov_ns["class_names"],
        image_path=image_path,
        layer_name=layer_name,
        n_steps=n_steps,
        segment_start=segment_start,
        segment_end=segment_end,
        clear_every=clear_every,
    )
    cov_ns["save_cache"](cache_dir, image_path, payload)
    return {
        "cache_path": str(cache_path),
    }


def _run_covariance_variant_once(
    *,
    cov_ns,
    image_path,
    layer_name,
    n_steps,
    segment_start,
    segment_end,
    selection_mode,
    selection_top_k,
    exact_segment_scale=None,
    sparse_scale=None,
    allow_build_missing=False,
    clear_every=8,
):
    image_path = Path(image_path)
    cache_dir, cache_path = _covariance_cache_location(
        cov_ns,
        image_path,
        layer_name,
        n_steps,
        segment_start,
        segment_end,
    )
    cache_hit = cache_path.exists()
    if cache_hit:
        payload = cov_ns["load_cache"](cache_path)
    elif allow_build_missing:
        payload = cov_ns["compute_cache_for_image"](
            model=cov_ns["model"],
            class_names=cov_ns["class_names"],
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            segment_start=segment_start,
            segment_end=segment_end,
            clear_every=clear_every,
        )
        cov_ns["save_cache"](cache_dir, image_path, payload)
    else:
        raise FileNotFoundError(f"Missing covariance cache: {cache_path}")

    method_vectors = covariance_sparse_methods_from_cache(
        cov_ns=cov_ns,
        cache=payload,
        selection_mode=selection_mode,
        selection_top_k=selection_top_k,
        exact_segment_scale=exact_segment_scale,
        sparse_scale=sparse_scale,
    )
    return {
        "cache_path": str(cache_path),
        "cache_hit": cache_hit,
        "vector_sum": float(np.sum(method_vectors["topk_exact_only_segment_scaled"])),
    }


def benchmark_covariance_sparse_summary_cached(
    *,
    image_paths,
    cov_ns,
    layer_name,
    n_steps,
    segment_start,
    segment_end,
    selection_mode,
    selection_top_k,
    train_n,
    repeats,
    clear_every,
    cache_root,
    target_device=None,
    refresh=False,
    clear_all_caches_fn=None,
    clear_covariance_backend_cache_fn=None,
    image_dir=None,
):
    current_device = current_device_label(cov_ns.get("DEVICE"))

    def clear_all():
        if clear_all_caches_fn is not None:
            clear_all_caches_fn()
        if clear_covariance_backend_cache_fn is not None:
            clear_covariance_backend_cache_fn()

    prepare_rows = []
    cache_paths = []
    for image_path in image_paths:
        image_key = image_signature(image_path)
        prepare_timing = time_call_cached(
            cache_root=cache_root,
            namespace="covariance_prepare_cache",
            config={
                "image": image_key,
                "layer_name": layer_name,
                "n_steps": int(n_steps),
                "segment_start": float(segment_start),
                "segment_end": float(segment_end),
                "clear_every": int(clear_every),
                "target_device": target_device,
            },
            label=f"{image_label(image_path)}_prepare",
            fn=lambda image_path=image_path: _run_covariance_prepare_once(
                cov_ns=cov_ns,
                image_path=image_path,
                layer_name=layer_name,
                n_steps=n_steps,
                segment_start=segment_start,
                segment_end=segment_end,
                clear_every=clear_every,
            ),
            repeats=repeats,
            refresh=refresh,
            clear_before=clear_all,
            required_device=target_device,
            current_device=current_device,
        )
        result = prepare_timing["result"] or {}
        cache_path = result.get("cache_path")
        if cache_path:
            cache_paths.append(cache_path)
        prepare_rows.append(
            {
                "image_path": image_path,
                "image_label": image_label(image_path),
                "time_s": prepare_timing["mean_s"],
                "runs": prepare_timing["runs"],
                "error": prepare_timing["error"],
                "cache_path": cache_path,
                "from_cache": prepare_timing["from_cache"],
            }
        )

    train_n = max(1, min(int(train_n), len(cache_paths)))
    scale_record = load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="covariance_scale_fit",
        config={
            "layer_name": layer_name,
            "n_steps": int(n_steps),
            "segment_start": float(segment_start),
            "segment_end": float(segment_end),
            "selection_mode": selection_mode,
            "selection_top_k": int(selection_top_k),
            "train_images": [image_signature(path) for path in image_paths[:train_n]],
            "target_device": target_device,
        },
        label=f"scale_{_selection_label(selection_mode, selection_top_k)}",
        compute_fn=lambda: fit_covariance_sparse_scales_from_cache_paths(
            cov_ns=cov_ns,
            cache_paths=cache_paths[:train_n],
            selection_mode=selection_mode,
            selection_top_k=selection_top_k,
        ),
        refresh=refresh,
        required_device=target_device,
        current_device=current_device,
    )
    scales = scale_record["value"] or {"exact_segment_scale": None, "sparse_scale": None}

    variant_rows = []
    for image_path in image_paths:
        image_key = image_signature(image_path)
        variant_timing = time_call_cached(
            cache_root=cache_root,
            namespace="covariance_variant",
            config={
                "image": image_key,
                "layer_name": layer_name,
                "n_steps": int(n_steps),
                "segment_start": float(segment_start),
                "segment_end": float(segment_end),
                "selection_mode": selection_mode,
                "selection_top_k": int(selection_top_k),
                "target_device": target_device,
            },
            label=f"{image_label(image_path)}_{_selection_label(selection_mode, selection_top_k)}",
            fn=lambda image_path=image_path: _run_covariance_variant_once(
                cov_ns=cov_ns,
                image_path=image_path,
                layer_name=layer_name,
                n_steps=n_steps,
                segment_start=segment_start,
                segment_end=segment_end,
                selection_mode=selection_mode,
                selection_top_k=selection_top_k,
                exact_segment_scale=scales.get("exact_segment_scale"),
                sparse_scale=scales.get("sparse_scale"),
                allow_build_missing=False,
                clear_every=clear_every,
            ),
            repeats=repeats,
            refresh=refresh,
            clear_before=clear_all,
            required_device=target_device,
            current_device=current_device,
        )
        result = variant_timing["result"] or {}
        variant_rows.append(
            {
                "image_path": image_path,
                "image_label": image_label(image_path),
                "time_s": variant_timing["mean_s"],
                "runs": variant_timing["runs"],
                "error": variant_timing["error"],
                "cache_path": result.get("cache_path"),
                "cache_hit": result.get("cache_hit"),
                "from_cache": variant_timing["from_cache"],
            }
        )

    comparison_rows = []
    n_images = len(image_paths)
    scale_fit_share_s = (scale_record["duration_s"] / n_images) if n_images else None
    for prepare_row, variant_row in zip(prepare_rows, variant_rows):
        total_s = None
        total_error = prepare_row["error"] or variant_row["error"] or scale_record["error"]
        if total_error is None:
            total_s = prepare_row["time_s"] + variant_row["time_s"] + scale_fit_share_s
        comparison_rows.append(
            {
                "image_path": prepare_row["image_path"],
                "image_label": prepare_row["image_label"],
                "time_s": total_s,
                "error": total_error,
            }
        )

    return {
        "kind": "classifier_covariance",
        "image_dir": str(image_dir) if image_dir is not None else None,
        "image_paths": list(image_paths),
        "layer_name": layer_name,
        "n_steps": n_steps,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "selection_mode": selection_mode,
        "selection_top_k": selection_top_k,
        "train_n": train_n,
        "repeats": repeats,
        "target_device": target_device,
        "cache_root": str(cache_root),
        "actual_device": current_device,
        "scales": scales,
        "scale_fit_s": scale_record["duration_s"],
        "scale_fit_error": scale_record["error"],
        "scale_fit_from_cache": scale_record["from_cache"],
        "prepare_rows": prepare_rows,
        "variant_rows": variant_rows,
        "comparison_rows": comparison_rows,
    }
