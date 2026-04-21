from __future__ import annotations

"""Unified NAOPC benchmark for classifier and detector attribution methods."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import IG, IG_det, NAA, NAA_det, cheap_ig
from modules.method_timing_cache import current_device_label, image_signature, load_or_compute_cached_value


DEFAULT_CACHE_ROOT = "output/naopc_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_CANDIDATE_TOP_K = 10
DEFAULT_REPORT_FILENAME = "naopc_report.md"
DEFAULT_SUMMARY_JSON = "naopc_summary.json"
DEFAULT_EPS = 1e-12
DEFAULT_UNIT_MODE = "spatial_cell"
DEFAULT_LIMIT_MODE = "beam"
DEFAULT_BEAM_SIZE = 5


def classifier_method_spec(kind, name=None, **kwargs):
    spec = {"kind": str(kind)}
    spec.update(kwargs)
    spec["name"] = str(name) if name is not None else _default_method_name(spec)
    return spec


def detector_method_spec(kind, name=None, **kwargs):
    spec = {"kind": str(kind)}
    spec.update(kwargs)
    spec["name"] = str(name) if name is not None else _default_method_name(spec)
    return spec


def default_classifier_method_specs(cheap_ig_variants=None):
    specs = [
        classifier_method_spec("ig", name="IG"),
        classifier_method_spec("naa", name="NAA"),
    ]
    for variant in cheap_ig_variants or []:
        specs.append(classifier_method_spec("cheap_ig", **variant))
    return specs


def default_detector_method_specs(cheap_ig_variants=None):
    specs = [
        detector_method_spec("ig", name="IG"),
        detector_method_spec("naa", name="NAA"),
    ]
    for variant in cheap_ig_variants or []:
        specs.append(detector_method_spec("cheap_ig", **variant))
    return specs


def benchmark_classifier_naopc(
    *,
    image_paths,
    method_specs,
    layer_name,
    n_steps,
    unit_mode=DEFAULT_UNIT_MODE,
    candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
    limit_mode=DEFAULT_LIMIT_MODE,
    beam_size=DEFAULT_BEAM_SIZE,
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
    verbose=False,
):
    return _run_naopc_benchmark(
        task="classifier",
        image_paths=image_paths,
        method_specs=method_specs,
        layer_name=layer_name,
        n_steps=n_steps,
        unit_mode=unit_mode,
        candidate_top_k=candidate_top_k,
        limit_mode=limit_mode,
        beam_size=beam_size,
        output_dir=output_dir,
        cache_root=cache_root,
        save_output=save_output,
        target_dir=target_dir,
        report_filename=report_filename,
        summary_filename=summary_filename,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
        verbose=verbose,
        runner_kwargs={
            "top_n": top_n,
            "fd_eps": fd_eps,
            "clear_every": clear_every,
        },
    )


def benchmark_detector_naopc(
    *,
    image_paths,
    method_specs,
    layer_name,
    n_steps,
    unit_mode=DEFAULT_UNIT_MODE,
    candidate_top_k=DEFAULT_CANDIDATE_TOP_K,
    limit_mode=DEFAULT_LIMIT_MODE,
    beam_size=DEFAULT_BEAM_SIZE,
    output_dir=DEFAULT_OUTPUT_DIR,
    cache_root=DEFAULT_CACHE_ROOT,
    save_output=True,
    target_dir=None,
    report_filename=DEFAULT_REPORT_FILENAME,
    summary_filename=DEFAULT_SUMMARY_JSON,
    mode="fixed_roi_mean",
    top_n=0,
    roi_top_k=-1,
    query_rank=None,
    query_head=None,
    bbox_iou_threshold=IG_det.BBOX_RANK_IOU_THRESHOLD,
    fd_eps=1e-3,
    clear_every=8,
    refresh_core=False,
    refresh_methods=False,
    verbose=False,
):
    return _run_naopc_benchmark(
        task="detector",
        image_paths=image_paths,
        method_specs=method_specs,
        layer_name=layer_name,
        n_steps=n_steps,
        unit_mode=unit_mode,
        candidate_top_k=candidate_top_k,
        limit_mode=limit_mode,
        beam_size=beam_size,
        output_dir=output_dir,
        cache_root=cache_root,
        save_output=save_output,
        target_dir=target_dir,
        report_filename=report_filename,
        summary_filename=summary_filename,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
        verbose=verbose,
        runner_kwargs={
            "mode": mode,
            "top_n": top_n,
            "roi_top_k": roi_top_k,
            "query_rank": query_rank,
            "query_head": query_head,
            "bbox_iou_threshold": bbox_iou_threshold,
            "fd_eps": fd_eps,
            "clear_every": clear_every,
        },
    )


def _run_naopc_benchmark(
    *,
    task,
    image_paths,
    method_specs,
    layer_name,
    n_steps,
    unit_mode,
    candidate_top_k,
    limit_mode,
    beam_size,
    output_dir,
    cache_root,
    save_output,
    target_dir,
    report_filename,
    summary_filename,
    refresh_core,
    refresh_methods,
    verbose,
    runner_kwargs,
):
    image_paths = [str(Path(path)) for path in image_paths]
    unit_mode = _normalize_unit_mode(unit_mode)
    limit_mode = _normalize_limit_mode(limit_mode)
    beam_size = _normalize_beam_size(beam_size, limit_mode)
    normalized_method_specs = _normalize_method_specs(task, method_specs)
    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = _prepare_existing_output_dir(target_dir)
        else:
            run_name = (
                f"{task}_naopc_{_safe_slug(unit_mode)}_{_safe_slug(layer_name)}_steps_{int(n_steps)}"
                f"_candidates_{int(candidate_top_k)}_limit_{_safe_slug(limit_mode)}"
            )
            if beam_size is not None:
                run_name += f"_beam_{int(beam_size)}"
            run_dir = _prepare_output_dir(output_dir, run_name)

    core_rows = []
    rows = []
    per_image = []

    for image_path in image_paths:
        core_record = _load_core_record(
            task=task,
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            unit_mode=unit_mode,
            candidate_top_k=candidate_top_k,
            limit_mode=limit_mode,
            beam_size=beam_size,
            cache_root=cache_root,
            refresh=refresh_core,
            verbose=verbose,
            runner_kwargs=runner_kwargs,
        )
        if core_record.get("error") is not None or core_record.get("value") is None:
            raise RuntimeError(
                f"Failed to build NAOPC core for {image_path}: {core_record.get('error')} "
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
                task=task,
                image_path=image_path,
                layer_name=layer_name,
                n_steps=n_steps,
                unit_mode=unit_mode,
                method_spec=method_spec,
                cache_root=cache_root,
                refresh=refresh_methods,
                verbose=verbose,
                runner_kwargs=runner_kwargs,
            )
            if method_record.get("error") is not None or method_record.get("value") is None:
                raise RuntimeError(
                    f"Failed to compute method {method_spec['name']} for {image_path}: {method_record.get('error')} "
                    f"(cache={method_record.get('cache_path')})"
                )
            method_value = method_record["value"]
            evaluation_record = _load_evaluation_record(
                task=task,
                image_path=image_path,
                layer_name=layer_name,
                n_steps=n_steps,
                unit_mode=unit_mode,
                limit_mode=limit_mode,
                beam_size=beam_size,
                method_spec=method_spec,
                core_value=core_value,
                method_value=method_value,
                cache_root=cache_root,
                refresh=refresh_methods,
                verbose=verbose,
                runner_kwargs=runner_kwargs,
            )
            if evaluation_record.get("error") is not None or evaluation_record.get("value") is None:
                raise RuntimeError(
                    f"Failed to evaluate method {method_spec['name']} for {image_path}: {evaluation_record.get('error')} "
                    f"(cache={evaluation_record.get('cache_path')})"
                )
            evaluation = evaluation_record["value"]
            row = {
                "task": task,
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
        task=task,
        rows=rows,
        core_rows=core_rows,
        method_specs=normalized_method_specs,
        layer_name=layer_name,
        n_steps=n_steps,
        unit_mode=unit_mode,
        candidate_top_k=candidate_top_k,
        limit_mode=limit_mode,
        beam_size=beam_size,
        cache_root=cache_root,
        runner_kwargs=runner_kwargs,
    )

    figures = {}
    if run_dir is not None:
        figures = _render_and_save_report_figures(
            run_dir=run_dir,
            summary=summary,
            rows=rows,
            core_rows=core_rows,
            method_specs=normalized_method_specs,
        )
        report_md = _build_report_markdown(summary, rows, core_rows, method_specs=normalized_method_specs, figures=figures)
        report_path = run_dir / report_filename
        report_path.write_text(report_md + "\n", encoding="utf-8")
        summary_path = run_dir / summary_filename
        summary_path.write_text(_pretty_json({"summary": summary, "rows": rows, "core_rows": core_rows}), encoding="utf-8")
    else:
        report_md = _build_report_markdown(summary, rows, core_rows, method_specs=normalized_method_specs, figures={})
        report_path = None
        summary_path = None

    return {
        "task": task,
        "image_paths": image_paths,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "unit_mode": unit_mode,
        "candidate_top_k": int(candidate_top_k),
        "limit_mode": limit_mode,
        "beam_size": beam_size,
        "cache_root": str(cache_root),
        "rows": rows,
        "core_rows": core_rows,
        "per_image": per_image,
        "method_specs": normalized_method_specs,
        "summary": summary,
        "report_markdown": report_md,
        "report_path": str(report_path) if report_path is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "figures": figures,
        "output_dir": str(run_dir) if run_dir is not None else None,
    }


def _load_core_record(
    *,
    task,
    image_path,
    layer_name,
    n_steps,
    unit_mode,
    candidate_top_k,
    limit_mode,
    beam_size,
    cache_root,
    refresh,
    verbose,
    runner_kwargs,
):
    image_key = image_signature(image_path)
    core_runner_kwargs = (
        {
            "mode": runner_kwargs["mode"],
            "roi_top_k": runner_kwargs["roi_top_k"],
            "query_rank": runner_kwargs["query_rank"],
            "query_head": runner_kwargs["query_head"],
            "bbox_iou_threshold": runner_kwargs["bbox_iou_threshold"],
        }
        if task == "detector"
        else {}
    )
    config = {
        "task": task,
        "image": image_key,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "unit_mode": unit_mode,
        "candidate_top_k": int(candidate_top_k),
        "limit_mode": limit_mode,
        "beam_size": beam_size,
        "runner_kwargs": core_runner_kwargs,
        "schema": 2,
    }
    namespace = f"naopc_{task}_core"
    if task == "classifier":
        compute_fn = lambda: _compute_classifier_core_value(
            image_path=image_path,
            layer_name=layer_name,
            unit_mode=unit_mode,
            candidate_top_k=candidate_top_k,
            limit_mode=limit_mode,
            beam_size=beam_size,
        )
    else:
        compute_fn = lambda: _compute_detector_core_value(
            image_path=image_path,
            layer_name=layer_name,
            unit_mode=unit_mode,
            candidate_top_k=candidate_top_k,
            limit_mode=limit_mode,
            beam_size=beam_size,
            mode=runner_kwargs["mode"],
            roi_top_k=runner_kwargs["roi_top_k"],
            query_rank=runner_kwargs["query_rank"],
            query_head=runner_kwargs["query_head"],
            bbox_iou_threshold=runner_kwargs["bbox_iou_threshold"],
        )
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace=namespace,
        config=config,
        label=Path(image_path).stem,
        compute_fn=compute_fn,
        refresh=refresh,
        required_device=None,
        current_device=_current_task_device_label(task),
    )


def _load_evaluation_record(
    *,
    task,
    image_path,
    layer_name,
    n_steps,
    unit_mode,
    limit_mode,
    beam_size,
    method_spec,
    core_value,
    method_value,
    cache_root,
    refresh,
    verbose,
    runner_kwargs,
):
    image_key = image_signature(image_path)
    evaluation_runner_kwargs = (
        {
            "mode": runner_kwargs["mode"],
            "roi_top_k": runner_kwargs["roi_top_k"],
            "query_rank": runner_kwargs["query_rank"],
            "query_head": runner_kwargs["query_head"],
            "bbox_iou_threshold": runner_kwargs["bbox_iou_threshold"],
        }
        if task == "detector"
        else {}
    )
    config = {
        "task": task,
        "image": image_key,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "unit_mode": unit_mode,
        "limit_mode": limit_mode,
        "beam_size": beam_size,
        "method_spec": method_spec,
        "candidate_unit_indices": core_value["candidate_unit_indices"],
        "runner_kwargs": evaluation_runner_kwargs,
        "schema": 1,
    }
    namespace = f"naopc_{task}_evaluation"
    if task == "classifier":
        compute_fn = lambda: _compute_classifier_evaluation_value(
            image_path=image_path,
            layer_name=layer_name,
            unit_mode=unit_mode,
            core_value=core_value,
            method_value=method_value,
        )
    else:
        compute_fn = lambda: _compute_detector_evaluation_value(
            image_path=image_path,
            layer_name=layer_name,
            unit_mode=unit_mode,
            core_value=core_value,
            method_value=method_value,
            mode=runner_kwargs["mode"],
            roi_top_k=runner_kwargs["roi_top_k"],
            query_rank=runner_kwargs["query_rank"],
            query_head=runner_kwargs["query_head"],
            bbox_iou_threshold=runner_kwargs["bbox_iou_threshold"],
        )
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace=namespace,
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}_eval",
        compute_fn=compute_fn,
        refresh=refresh,
        required_device=None,
        current_device=_current_task_device_label(task),
    )


def _load_method_record(
    *,
    task,
    image_path,
    layer_name,
    n_steps,
    unit_mode,
    method_spec,
    cache_root,
    refresh,
    verbose,
    runner_kwargs,
):
    image_key = image_signature(image_path)
    config = {
        "task": task,
        "image": image_key,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "unit_mode": unit_mode,
        "method_spec": method_spec,
        "runner_kwargs": runner_kwargs,
        "schema": 1,
    }
    namespace = f"naopc_{task}_method"
    if task == "classifier":
        compute_fn = lambda: _compute_classifier_method_value(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            unit_mode=unit_mode,
            method_spec=method_spec,
            fd_eps=runner_kwargs["fd_eps"],
            clear_every=runner_kwargs["clear_every"],
        )
    else:
        compute_fn = lambda: _compute_detector_method_value(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            unit_mode=unit_mode,
            method_spec=method_spec,
            mode=runner_kwargs["mode"],
            roi_top_k=runner_kwargs["roi_top_k"],
            query_rank=runner_kwargs["query_rank"],
            query_head=runner_kwargs["query_head"],
            bbox_iou_threshold=runner_kwargs["bbox_iou_threshold"],
            fd_eps=runner_kwargs["fd_eps"],
            clear_every=runner_kwargs["clear_every"],
        )
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace=namespace,
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        compute_fn=compute_fn,
        refresh=refresh,
        required_device=None,
        current_device=_current_task_device_label(task),
    )


def _compute_classifier_core_value(*, image_path, layer_name, unit_mode, candidate_top_k, limit_mode, beam_size):
    _clear_all_backend_caches()
    runner = _ClassifierPerturbationRunner(image_path=image_path, layer_name=layer_name, unit_mode=unit_mode)
    try:
        candidate_indices, candidate_scores = runner.select_candidate_units(candidate_top_k)
        (
            subset_drops,
            lower_limit,
            upper_limit,
            lower_order,
            upper_order,
            lower_curve,
            upper_curve,
        ) = _compute_naopc_limits(
            runner=runner,
            candidate_indices=candidate_indices,
            limit_mode=limit_mode,
            beam_size=beam_size,
        )
        return {
            "task": "classifier",
            "image_path": image_path,
            "layer_name": layer_name,
            "unit_mode": unit_mode,
            "limit_mode": limit_mode,
            "beam_size": beam_size,
            "target_class": int(runner.target_class),
            "target_name": runner.target_name,
            "clean_score": float(runner.clean_score),
            "baseline_score": float(runner.baseline_score),
            "clean_delta": float(runner.clean_score - runner.baseline_score),
            "candidate_unit_indices": [int(v) for v in candidate_indices.tolist()],
            "candidate_reference_scores": [float(v) for v in candidate_scores.tolist()],
            "candidate_unit_labels": [runner.unit_labels[idx] for idx in candidate_indices.tolist()],
            "subset_drops": [float(v) for v in subset_drops.tolist()] if subset_drops is not None else None,
            "lower_limit": float(lower_limit),
            "upper_limit": float(upper_limit),
            "lower_order_positions": [int(v) for v in lower_order.tolist()],
            "upper_order_positions": [int(v) for v in upper_order.tolist()],
            "lower_curve": [float(v) for v in lower_curve.tolist()],
            "upper_curve": [float(v) for v in upper_curve.tolist()],
            "n_units_total": int(runner.n_units),
            "candidate_top_k": int(len(candidate_indices)),
            "subset_eval_count": int(len(runner._drop_cache) - 1),
        }
    finally:
        runner.close()
        _clear_all_backend_caches()


def _compute_detector_core_value(
    *,
    image_path,
    layer_name,
    unit_mode,
    candidate_top_k,
    limit_mode,
    beam_size,
    mode,
    roi_top_k,
    query_rank,
    query_head,
    bbox_iou_threshold,
):
    _clear_all_backend_caches()
    runner = _DetectorPerturbationRunner(
        image_path=image_path,
        layer_name=layer_name,
        unit_mode=unit_mode,
        mode=mode,
        roi_top_k=roi_top_k,
        query_rank=query_rank,
        query_head=query_head,
        bbox_iou_threshold=bbox_iou_threshold,
    )
    try:
        candidate_indices, candidate_scores = runner.select_candidate_units(candidate_top_k)
        (
            subset_drops,
            lower_limit,
            upper_limit,
            lower_order,
            upper_order,
            lower_curve,
            upper_curve,
        ) = _compute_naopc_limits(
            runner=runner,
            candidate_indices=candidate_indices,
            limit_mode=limit_mode,
            beam_size=beam_size,
        )
        target_spec = _normalize_target_spec(runner.target_spec)
        return {
            "task": "detector",
            "image_path": image_path,
            "layer_name": layer_name,
            "unit_mode": unit_mode,
            "limit_mode": limit_mode,
            "beam_size": beam_size,
            "resolved_layer_names": list(runner.resolved_layer_names),
            "mode": mode,
            "target_class": int(target_spec["class_index"]),
            "target_name": runner.target_name,
            "target_spec": target_spec,
            "clean_score": float(runner.clean_score),
            "baseline_score": float(runner.baseline_score),
            "clean_delta": float(runner.clean_score - runner.baseline_score),
            "candidate_unit_indices": [int(v) for v in candidate_indices.tolist()],
            "candidate_reference_scores": [float(v) for v in candidate_scores.tolist()],
            "candidate_unit_labels": [runner.unit_labels[idx] for idx in candidate_indices.tolist()],
            "subset_drops": [float(v) for v in subset_drops.tolist()] if subset_drops is not None else None,
            "lower_limit": float(lower_limit),
            "upper_limit": float(upper_limit),
            "lower_order_positions": [int(v) for v in lower_order.tolist()],
            "upper_order_positions": [int(v) for v in upper_order.tolist()],
            "lower_curve": [float(v) for v in lower_curve.tolist()],
            "upper_curve": [float(v) for v in upper_curve.tolist()],
            "n_units_total": int(runner.n_units),
            "candidate_top_k": int(len(candidate_indices)),
            "subset_eval_count": int(len(runner._drop_cache) - 1),
            "roi_top_k": int(roi_top_k),
            "query_rank": query_rank,
            "query_head": query_head,
            "bbox_iou_threshold": float(bbox_iou_threshold),
        }
    finally:
        runner.close()
        _clear_all_backend_caches()


def _compute_classifier_method_value(*, image_path, layer_name, n_steps, unit_mode, method_spec, fd_eps, clear_every):
    _clear_all_backend_caches()
    result = _run_classifier_method(
        image_path=image_path,
        layer_name=layer_name,
        n_steps=n_steps,
        unit_mode=unit_mode,
        method_spec=method_spec,
        fd_eps=fd_eps,
        clear_every=clear_every,
    )
    _clear_all_backend_caches()
    return result


def _compute_classifier_evaluation_value(*, image_path, layer_name, unit_mode, core_value, method_value):
    subset_drops = core_value.get("subset_drops")
    if subset_drops is not None:
        return _evaluate_method_against_core(core_value, method_value)

    _clear_all_backend_caches()
    runner = _ClassifierPerturbationRunner(
        image_path=image_path,
        layer_name=layer_name,
        unit_mode=unit_mode,
        target_class=core_value.get("target_class"),
    )
    try:
        return _evaluate_method_against_core_with_runner(core_value, method_value, runner)
    finally:
        runner.close()
        _clear_all_backend_caches()


def _compute_detector_method_value(
    *,
    image_path,
    layer_name,
    n_steps,
    unit_mode,
    method_spec,
    mode,
    roi_top_k,
    query_rank,
    query_head,
    bbox_iou_threshold,
    fd_eps,
    clear_every,
):
    _clear_all_backend_caches()
    result = _run_detector_method(
        image_path=image_path,
        layer_name=layer_name,
        n_steps=n_steps,
        unit_mode=unit_mode,
        method_spec=method_spec,
        mode=mode,
        roi_top_k=roi_top_k,
        query_rank=query_rank,
        query_head=query_head,
        bbox_iou_threshold=bbox_iou_threshold,
        fd_eps=fd_eps,
        clear_every=clear_every,
    )
    _clear_all_backend_caches()
    return result


def _compute_detector_evaluation_value(
    *,
    image_path,
    layer_name,
    unit_mode,
    core_value,
    method_value,
    mode,
    roi_top_k,
    query_rank,
    query_head,
    bbox_iou_threshold,
):
    subset_drops = core_value.get("subset_drops")
    if subset_drops is not None:
        return _evaluate_method_against_core(core_value, method_value)

    _clear_all_backend_caches()
    runner = _DetectorPerturbationRunner(
        image_path=image_path,
        layer_name=layer_name,
        unit_mode=unit_mode,
        mode=mode,
        roi_top_k=roi_top_k,
        query_rank=query_rank,
        query_head=query_head,
        bbox_iou_threshold=bbox_iou_threshold,
        target_spec=core_value.get("target_spec"),
    )
    try:
        return _evaluate_method_against_core_with_runner(core_value, method_value, runner)
    finally:
        runner.close()
        _clear_all_backend_caches()


def _run_classifier_method(*, image_path, layer_name, n_steps, unit_mode, method_spec, fd_eps, clear_every):
    kind = method_spec["kind"]
    common = {
        "image_path": image_path,
        "layer_name": layer_name,
        "n_steps": n_steps,
        "top_n": 0,
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
            **common,
        )
    else:
        raise ValueError(f"Unsupported classifier method kind: {kind}")
    return _serialize_method_payload(payload, method_spec, unit_mode=unit_mode)


def _run_detector_method(
    *,
    image_path,
    layer_name,
    n_steps,
    unit_mode,
    method_spec,
    mode,
    roi_top_k,
    query_rank,
    query_head,
    bbox_iou_threshold,
    fd_eps,
    clear_every,
):
    kind = method_spec["kind"]
    common = {
        "image_path": image_path,
        "layer_name": layer_name,
        "mode": mode,
        "n_steps": n_steps,
        "top_n": 0,
        "roi_top_k": roi_top_k,
        "query_rank": query_rank,
        "query_head": query_head,
        "bbox_iou_threshold": bbox_iou_threshold,
        "clear_every": clear_every,
        "verbose": False,
        "show_total_plot": False,
        "show_filter_plots": False,
        "show_target_box": False,
    }
    if kind == "ig":
        payload = IG_det.run_detector_conductance(fd_eps=fd_eps, **common)
    elif kind == "naa":
        if mode == "fixed_query":
            raise ValueError("Detector NAA benchmark does not support mode='fixed_query'. Use a fixed ROI mode.")
        payload = NAA_det.run_attribution_pipeline(**common)
    elif kind == "cheap_ig":
        payload = cheap_ig.run_detector_cheap_ig_pipeline(
            segment_start=method_spec.get("segment_start", 0.0),
            segment_end=method_spec.get("segment_end", 0.1),
            selection_mode=method_spec.get("selection_mode", "signed"),
            selection_top_k=method_spec.get("selection_top_k", 5000),
            **common,
        )
    else:
        raise ValueError(f"Unsupported detector method kind: {kind}")
    return _serialize_method_payload(payload, method_spec, unit_mode=unit_mode)


def _serialize_method_payload(payload, method_spec, *, unit_mode):
    unit_scores = _unit_scores_from_payload(payload, unit_mode)
    value = {
        "method_name": method_spec["name"],
        "method_id": method_spec["id"],
        "kind": method_spec["kind"],
        "unit_mode": unit_mode,
        "unit_scores": [float(v) for v in unit_scores.tolist()],
        "abs_error": float(payload.get("abs_error", float("nan"))),
        "fx": float(payload.get("fx", float("nan"))),
        "fx0": float(payload.get("fx0", float("nan"))),
    }
    for optional_key in (
        "selection_mode",
        "selection_top_k",
        "segment_start",
        "segment_end",
        "selected_neurons",
        "roi_query_count",
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


class _ClassifierPerturbationRunner:
    def __init__(self, *, image_path, layer_name, unit_mode, target_class=None):
        self.image_path = str(image_path)
        self.layer_name = layer_name
        self.unit_mode = _normalize_unit_mode(unit_mode)
        self.x, _ = IG.load_image(self.image_path)
        self.x0 = IG.black_baseline_like(self.x)
        self.capture_hook = IG.LayerHook(IG.model, self.layer_name)
        try:
            clean_out, clean_act = self._forward_with_capture(self.x)
            base_out, base_act = self._forward_with_capture(self.x0)
            _, clean_logits = IG.split_classifier_output(clean_out)
            _, base_logits = IG.split_classifier_output(base_out)
            if target_class is None:
                self.target_class = int(clean_logits[0].argmax().item())
            else:
                self.target_class = int(target_class)
            self.target_name = IG.class_names[self.target_class]
            self.clean_score = float(clean_logits[0, self.target_class].item())
            self.baseline_score = float(base_logits[0, self.target_class].item())
            self.clean_act = clean_act.detach()
            self.base_act = base_act.detach()
        finally:
            self.capture_hook.remove()

        self.unit_labels = _build_single_tensor_unit_labels(
            self.layer_name,
            self.clean_act.shape,
            unit_mode=self.unit_mode,
        )
        self.n_units = int(len(self.unit_labels))
        self._drop_cache = {0: 0.0}

    def _forward_with_capture(self, x_in):
        self.capture_hook.clear()
        out = IG.model(x_in)
        act = IG.unwrap_tensor(self.capture_hook.get())
        return out, act

    def select_candidate_units(self, candidate_top_k):
        reference = _unit_reference_scores_from_tensor(self.clean_act - self.base_act, unit_mode=self.unit_mode)
        candidate_top_k = max(1, min(int(candidate_top_k), reference.size))
        order = np.argsort(-reference, kind="stable")[:candidate_top_k]
        return order.astype(np.int64), reference[order].astype(np.float64, copy=False)

    def score_drop_for_unit_indices(self, unit_indices):
        unit_indices = tuple(sorted(int(v) for v in unit_indices))
        if not unit_indices:
            return 0.0
        if unit_indices in self._drop_cache:
            return self._drop_cache[unit_indices]

        baseline = self.base_act
        index_tensor = torch.tensor(unit_indices, device=baseline.device, dtype=torch.long)
        modules = dict(IG.model.named_modules())
        handle = modules[self.layer_name].register_forward_hook(
            lambda module, inp, out: _patch_single_tensor_units(
                out,
                baseline,
                index_tensor,
                unit_mode=self.unit_mode,
            )
        )
        try:
            with torch.no_grad():
                out = IG.model(self.x)
                _, logits = IG.split_classifier_output(out)
                patched_score = float(logits[0, self.target_class].item())
            drop = float(self.clean_score - patched_score)
        finally:
            handle.remove()
        self._drop_cache[unit_indices] = drop
        return drop

    def close(self):
        pass


class _DetectorPerturbationRunner:
    def __init__(
        self,
        *,
        image_path,
        layer_name,
        unit_mode,
        mode,
        roi_top_k,
        query_rank,
        query_head,
        bbox_iou_threshold,
        target_spec=None,
    ):
        self.image_path = str(image_path)
        self.layer_name = layer_name
        self.unit_mode = _normalize_unit_mode(unit_mode)
        self.mode = mode
        self.roi_top_k = roi_top_k
        self.query_rank = query_rank
        self.query_head = query_head
        self.bbox_iou_threshold = bbox_iou_threshold

        if layer_name in IG_det.LAYER_GROUPS:
            self.resolved_layer_names = tuple(IG_det.LAYER_GROUPS[layer_name])
        else:
            self.resolved_layer_names = (layer_name,)

        self.x, _, _ = IG_det.load_image(self.image_path)
        self.x0 = IG_det.black_baseline_like(self.x)
        self.capture_hook = IG_det.LayerHook(IG_det.model, self.resolved_layer_names)
        try:
            clean_out, clean_act = self._forward_with_capture(self.x)
            base_out, base_act = self._forward_with_capture(self.x0)
            self.target_spec = (
                _resolve_detector_target_spec(
                    raw_output=clean_out,
                    mode=mode,
                    roi_top_k=roi_top_k,
                    query_rank=query_rank,
                    query_head=query_head,
                    bbox_iou_threshold=bbox_iou_threshold,
                )
                if target_spec is None
                else _normalize_target_spec(target_spec)
            )
            self.clean_score = float(IG_det.detection_scalar_target(clean_out, self.target_spec, len(IG_det.class_names)).item())
            self.baseline_score = float(
                IG_det.detection_scalar_target(base_out, self.target_spec, len(IG_det.class_names)).item()
            )
            self.clean_parts = _to_part_tuple(clean_act)
            self.base_parts = _to_part_tuple(base_act)
        finally:
            self.capture_hook.remove()

        self.part_unit_counts = _part_unit_counts(self.clean_parts, unit_mode=self.unit_mode)
        self.unit_labels = _build_multi_part_unit_labels(
            self.resolved_layer_names,
            self.clean_parts,
            unit_mode=self.unit_mode,
        )
        self.n_units = int(len(self.unit_labels))
        self.target_name = IG_det.class_names[int(self.target_spec["class_index"])]
        self._drop_cache = {0: 0.0}

    def _forward_with_capture(self, x_in):
        self.capture_hook.clear()
        out = IG_det.model(x_in)
        act = IG_det.unwrap_tensor(self.capture_hook.get())
        return out, act

    def select_candidate_units(self, candidate_top_k):
        reference = _unit_reference_scores_from_parts(
            tuple(clean - base for clean, base in zip(self.clean_parts, self.base_parts))
            ,
            unit_mode=self.unit_mode,
        )
        candidate_top_k = max(1, min(int(candidate_top_k), reference.size))
        order = np.argsort(-reference, kind="stable")[:candidate_top_k]
        return order.astype(np.int64), reference[order].astype(np.float64, copy=False)

    def score_drop_for_unit_indices(self, unit_indices):
        unit_indices = tuple(sorted(int(v) for v in unit_indices))
        if not unit_indices:
            return 0.0
        if unit_indices in self._drop_cache:
            return self._drop_cache[unit_indices]

        per_part = _split_flat_unit_indices(unit_indices, self.part_unit_counts)
        modules = dict(IG_det.model.named_modules())
        handles = []
        try:
            for part_idx, local_indices in enumerate(per_part):
                if not local_indices:
                    continue
                baseline = self.base_parts[part_idx]
                index_tensor = torch.tensor(local_indices, device=baseline.device, dtype=torch.long)
                layer_name = self.resolved_layer_names[part_idx]
                handles.append(
                    modules[layer_name].register_forward_hook(
                        lambda module, inp, out, baseline=baseline, index_tensor=index_tensor: _patch_single_tensor_units(
                            out,
                            baseline,
                            index_tensor,
                            unit_mode=self.unit_mode,
                        )
                    )
                )
            with torch.no_grad():
                raw_out = IG_det.model(self.x)
                patched_score = float(
                    IG_det.detection_scalar_target(raw_out, self.target_spec, len(IG_det.class_names)).item()
                )
            drop = float(self.clean_score - patched_score)
        finally:
            for handle in handles:
                handle.remove()
        self._drop_cache[unit_indices] = drop
        return drop

    def close(self):
        pass


def _resolve_detector_target_spec(raw_output, mode, roi_top_k, query_rank, query_head, bbox_iou_threshold):
    if mode == "fixed_query":
        return IG_det.pick_fixed_query_target(
            raw_output,
            num_classes=len(IG_det.class_names),
            query_rank=query_rank,
            query_head=query_head,
            bbox_iou_threshold=bbox_iou_threshold,
        )
    return IG_det.pick_fixed_roi_target(
        raw_output,
        num_classes=len(IG_det.class_names),
        roi_mode=mode,
        roi_top_k=roi_top_k,
        query_rank=query_rank,
        query_head=query_head,
        bbox_iou_threshold=bbox_iou_threshold,
    )


def _evaluate_method_against_core(core_value, method_value):
    subset_drops = np.asarray(core_value["subset_drops"], dtype=np.float64)
    candidate_indices, candidate_method_scores, order_positions = _resolve_method_order(core_value, method_value)
    curve = _curve_from_order(subset_drops, order_positions)
    return _finalize_method_evaluation(
        core_value=core_value,
        method_value=method_value,
        candidate_indices=candidate_indices,
        candidate_method_scores=candidate_method_scores,
        order_positions=order_positions,
        curve=curve,
    )


def _evaluate_method_against_core_with_runner(core_value, method_value, runner):
    candidate_indices, candidate_method_scores, order_positions = _resolve_method_order(core_value, method_value)
    curve = _curve_from_order_runner(runner, candidate_indices, order_positions)
    return _finalize_method_evaluation(
        core_value=core_value,
        method_value=method_value,
        candidate_indices=candidate_indices,
        candidate_method_scores=candidate_method_scores,
        order_positions=order_positions,
        curve=curve,
    )


def _resolve_method_order(core_value, method_value):
    unit_scores = np.asarray(method_value["unit_scores"], dtype=np.float64)
    candidate_indices = np.asarray(core_value["candidate_unit_indices"], dtype=np.int64)
    if candidate_indices.size == 0:
        raise ValueError("NAOPC benchmark requires at least one candidate unit.")
    if unit_scores.size <= int(candidate_indices.max()):
        raise ValueError(
            f"Method unit_scores has size={unit_scores.size}, but candidate benchmark expects index {int(candidate_indices.max())}."
        )

    candidate_method_scores = unit_scores[candidate_indices]
    candidate_reference = np.asarray(core_value["candidate_reference_scores"], dtype=np.float64)
    order_positions = _stable_descending_order(candidate_method_scores, candidate_reference, candidate_indices)
    return candidate_indices, candidate_method_scores, order_positions


def _finalize_method_evaluation(*, core_value, method_value, candidate_indices, candidate_method_scores, order_positions, curve):
    aopc = float(curve.mean())

    lower = float(core_value["lower_limit"])
    upper = float(core_value["upper_limit"])
    denom = upper - lower
    if abs(denom) <= DEFAULT_EPS:
        naopc = float("nan")
        naopc_clipped = float("nan")
    else:
        naopc = float((aopc - lower) / denom)
        naopc_clipped = float(np.clip(naopc, 0.0, 1.0))

    return {
        "aopc": aopc,
        "naopc": naopc,
        "naopc_clipped": naopc_clipped,
        "curve": [float(v) for v in curve.tolist()],
        "candidate_scores": [float(v) for v in candidate_method_scores.tolist()],
        "order_positions": [int(v) for v in order_positions.tolist()],
        "ordered_unit_labels": [core_value["candidate_unit_labels"][idx] for idx in order_positions.tolist()],
        "ordered_unit_indices": [int(candidate_indices[idx]) for idx in order_positions.tolist()],
        "abs_error": float(method_value.get("abs_error", float("nan"))),
        "fx": float(method_value.get("fx", float("nan"))),
        "fx0": float(method_value.get("fx0", float("nan"))),
        "selected_neurons": method_value.get("selected_neurons"),
    }


def _attach_method_ranks(rows):
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_path"], []).append(row)
    for image_rows in by_image.values():
        valid = [row for row in image_rows if row["naopc"] == row["naopc"]]
        valid.sort(key=lambda item: (-item["naopc"], item["method_name"]))
        rank_map = {row["method_name"]: index + 1 for index, row in enumerate(valid)}
        for row in image_rows:
            row["rank"] = rank_map.get(row["method_name"])


def _build_benchmark_summary(
    *,
    task,
    rows,
    core_rows,
    method_specs,
    layer_name,
    n_steps,
    unit_mode,
    candidate_top_k,
    limit_mode,
    beam_size,
    cache_root,
    runner_kwargs,
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
            "naopc": _stats_record([row["naopc"] for row in method_rows]),
            "naopc_clipped": _stats_record([row["naopc_clipped"] for row in method_rows]),
            "aopc": _stats_record([row["aopc"] for row in method_rows]),
            "abs_error": _stats_record([row["abs_error"] for row in method_rows]),
            "runtime_s": _stats_record([row["method_duration_s"] for row in method_rows]),
            "evaluation_runtime_s": _stats_record([row["evaluation_duration_s"] for row in method_rows]),
            "benchmark_runtime_s": _stats_record([row["benchmark_duration_s"] for row in method_rows]),
            "rank": _stats_record([row["rank"] for row in method_rows if row.get("rank") is not None]),
        }

    pairwise_win_rate = _pairwise_win_rate(rows, method_specs)
    core_summary = {
        "upper_limit": _stats_record([row["core"]["upper_limit"] for row in core_rows]),
        "lower_limit": _stats_record([row["core"]["lower_limit"] for row in core_rows]),
        "clean_delta": _stats_record([row["core"]["clean_delta"] for row in core_rows]),
        "subset_eval_count": _stats_record([row["core"].get("subset_eval_count") for row in core_rows]),
        "core_runtime_s": _stats_record([row["core_duration_s"] for row in core_rows]),
    }

    return {
        "task": task,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "unit_mode": unit_mode,
        "candidate_top_k": int(candidate_top_k),
        "limit_mode": limit_mode,
        "beam_size": beam_size,
        "n_images": int(len(core_rows)),
        "cache_root": str(cache_root),
        "runner_kwargs": runner_kwargs,
        "method_summaries": method_summaries,
        "core_summary": core_summary,
        "pairwise_win_rate": pairwise_win_rate,
    }


def _render_and_save_report_figures(*, run_dir, summary, rows, core_rows, method_specs):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    fig = _plot_naopc_summary(summary, method_specs)
    path = figure_dir / "naopc_summary.png"
    _save_figure(fig, path)
    figures["naopc_summary"] = str(path)

    fig = _plot_naopc_distributions(rows, method_specs)
    path = figure_dir / "naopc_distributions.png"
    _save_figure(fig, path)
    figures["naopc_distributions"] = str(path)

    fig = _plot_mean_curves(rows, core_rows, method_specs)
    path = figure_dir / "naopc_curves.png"
    _save_figure(fig, path)
    figures["naopc_curves"] = str(path)

    fig = _plot_pairwise_win_heatmap(summary, method_specs)
    path = figure_dir / "naopc_pairwise_wins.png"
    _save_figure(fig, path)
    figures["naopc_pairwise_wins"] = str(path)

    return figures


def _build_report_markdown(summary, rows, core_rows, *, method_specs, figures):
    lines = [
        "# NAOPC Benchmark",
        "",
        f"- task=`{summary['task']}`",
        f"- layer_name=`{summary['layer_name']}`",
        f"- n_steps={summary['n_steps']}",
        f"- unit_mode=`{summary['unit_mode']}`",
        f"- candidate_top_k={summary['candidate_top_k']}",
        f"- limit_mode=`{summary['limit_mode']}`",
        f"- beam_size={summary['beam_size']!r}",
        f"- n_images={summary['n_images']}",
        f"- cache_root=`{summary['cache_root']}`",
    ]

    runner_kwargs = summary.get("runner_kwargs") or {}
    for key in sorted(runner_kwargs):
        lines.append(f"- {key}={runner_kwargs[key]!r}")

    lines.extend(
        [
            "",
            "## Aggregate Summary",
            "",
            "| Method | NAOPC | AOPC | Mean Rank | Attr Runtime (s) | Eval Runtime (s) | Abs Error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method_spec in method_specs:
        stats = summary["method_summaries"][method_spec["name"]]
        lines.append(
            "| {method} | {naopc} | {aopc} | {rank} | {runtime} | {eval_runtime} | {abs_error} |".format(
                method=method_spec["name"],
                naopc=_format_stats(stats["naopc"]),
                aopc=_format_stats(stats["aopc"]),
                rank=_format_stats(stats["rank"]),
                runtime=_format_stats(stats["runtime_s"]),
                eval_runtime=_format_stats(stats["evaluation_runtime_s"]),
                abs_error=_format_stats(stats["abs_error"]),
            )
        )

    core_summary = summary["core_summary"]
    lines.extend(
        [
            "",
            "## Core Limits",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| upper_limit | {_format_stats(core_summary['upper_limit'])} |",
            f"| lower_limit | {_format_stats(core_summary['lower_limit'])} |",
            f"| clean_delta | {_format_stats(core_summary['clean_delta'])} |",
            f"| subset_eval_count | {_format_stats(core_summary['subset_eval_count'])} |",
            f"| core_runtime_s | {_format_stats(core_summary['core_runtime_s'])} |",
        ]
    )

    if figures:
        lines.extend(["", "## Figures", ""])
        for key in ("naopc_summary", "naopc_distributions", "naopc_curves", "naopc_pairwise_wins"):
            path = figures.get(key)
            if path:
                rel_path = _relative_markdown_path(path)
                lines.append(f"### {key}")
                lines.append("")
                lines.append(f"![]({rel_path})")
                lines.append("")

    lines.extend(
        [
            "## Per-Image NAOPC",
            "",
            _build_per_image_markdown_table(rows, method_specs),
            "",
            "## Pairwise Win Rate",
            "",
            _build_pairwise_markdown_table(summary["pairwise_win_rate"], method_specs),
        ]
    )
    return "\n".join(lines)


def _plot_naopc_summary(summary, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    means = [summary["method_summaries"][name]["naopc"]["mean"] for name in method_names]
    stds = [summary["method_summaries"][name]["naopc"]["std"] for name in method_names]
    ranks = [summary["method_summaries"][name]["rank"]["mean"] for name in method_names]
    x = np.arange(len(method_names))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(x, means, yerr=stds, capsize=6, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylabel("NAOPC")
    ax.set_title(
        "Mean NAOPC by method, "
        f"n={summary['n_images']}, unit={summary['unit_mode']}, { _limit_mode_label(summary['limit_mode'], summary['beam_size']) }"
    )
    ax.grid(axis="y", alpha=0.3)
    for idx, (mean_value, rank_value) in enumerate(zip(means, ranks)):
        if mean_value == mean_value:
            label = f"rank={rank_value:.2f}" if rank_value == rank_value else "rank=n/a"
            ax.text(idx, mean_value + max(0.01, stds[idx] * 0.3 + 0.01), label, ha="center", va="bottom", fontsize=9)
    return fig


def _plot_naopc_distributions(rows, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    values = []
    for name in method_names:
        current = [row["naopc"] for row in rows if row["method_name"] == name and row["naopc"] == row["naopc"]]
        values.append(current if current else [float("nan")])

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    box = ax.boxplot(values, patch_artist=True, labels=method_names)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(method_names))))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_ylabel("NAOPC")
    ax.set_title("NAOPC distribution by method")
    ax.grid(axis="y", alpha=0.3)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
    return fig


def _plot_mean_curves(rows, core_rows, method_specs):
    candidate_top_k = int(core_rows[0]["core"]["candidate_top_k"]) if core_rows else 0
    x = np.arange(1, candidate_top_k + 1)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)

    core_lower = np.asarray([row["core"]["lower_curve"] for row in core_rows], dtype=np.float64)
    core_upper = np.asarray([row["core"]["upper_curve"] for row in core_rows], dtype=np.float64)
    if core_lower.size > 0:
        axes[0].plot(x, core_lower.mean(axis=0), linestyle="--", color="black", label="lower limit")
        axes[0].plot(x, core_upper.mean(axis=0), linestyle=":", color="black", label="upper limit")
        axes[1].plot(x, core_lower.mean(axis=0), linestyle="--", color="black", label="lower limit")
        axes[1].plot(x, core_upper.mean(axis=0), linestyle=":", color="black", label="upper limit")

    for color_idx, spec in enumerate(method_specs):
        method_rows = [row for row in rows if row["method_name"] == spec["name"]]
        curves = np.asarray([row["curve"] for row in method_rows], dtype=np.float64)
        if curves.size == 0:
            continue
        color = plt.cm.tab10(color_idx % 10)
        axes[0].plot(x, curves.mean(axis=0), label=spec["name"], color=color)

        normalized_curves = []
        for row in method_rows:
            core = next(item["core"] for item in core_rows if item["image_path"] == row["image_path"])
            denom = float(core["clean_delta"])
            if abs(denom) <= DEFAULT_EPS:
                normalized_curves.append([float("nan")] * len(row["curve"]))
            else:
                normalized_curves.append([float(value) / denom for value in row["curve"]])
        normalized_curves = np.asarray(normalized_curves, dtype=np.float64)
        axes[1].plot(x, np.nanmean(normalized_curves, axis=0), label=spec["name"], color=color)

    unit_mode = core_rows[0]["core"].get("unit_mode", DEFAULT_UNIT_MODE) if core_rows else DEFAULT_UNIT_MODE
    limit_mode = core_rows[0]["core"].get("limit_mode", DEFAULT_LIMIT_MODE) if core_rows else DEFAULT_LIMIT_MODE
    beam_size = core_rows[0]["core"].get("beam_size", DEFAULT_BEAM_SIZE) if core_rows else DEFAULT_BEAM_SIZE
    limit_label = _limit_mode_label(limit_mode, beam_size)
    axes[0].set_title(f"Mean perturbation curve, unit={unit_mode}, {limit_label}")
    axes[0].set_xlabel("Perturbed candidate units")
    axes[0].set_ylabel("Score drop")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title(f"Mean perturbation curve / clean delta, unit={unit_mode}, {limit_label}")
    axes[1].set_xlabel("Perturbed candidate units")
    axes[1].set_ylabel("Normalized score drop")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
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
        by_image.setdefault(row["image_path"], {})[row["method_name"]] = row["naopc"]

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
        by_image.setdefault(row["image_name"], {})[row["method_name"]] = row["naopc"]
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


def _compute_subset_drops(runner, candidate_indices):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    subset_drops = np.zeros(1 << len(candidate_indices), dtype=np.float64)
    for mask in range(1, 1 << len(candidate_indices)):
        selected = [int(candidate_indices[pos]) for pos in range(len(candidate_indices)) if (mask >> pos) & 1]
        subset_drops[mask] = runner.score_drop_for_unit_indices(selected)
    return subset_drops


def _compute_naopc_limits(*, runner, candidate_indices, limit_mode, beam_size):
    n_candidates = int(len(candidate_indices))
    if n_candidates <= 0:
        raise ValueError("NAOPC benchmark requires at least one candidate unit.")

    if limit_mode == "exact":
        subset_drops = _compute_subset_drops(runner, candidate_indices)
        lower_limit, upper_limit, lower_order, upper_order = _solve_naopc_limits_exact(subset_drops, n_candidates)
        lower_curve = _curve_from_order(subset_drops, lower_order)
        upper_curve = _curve_from_order(subset_drops, upper_order)
        return subset_drops, lower_limit, upper_limit, lower_order, upper_order, lower_curve, upper_curve

    lower_limit, lower_order, lower_curve = _solve_naopc_limit_beam(
        runner=runner,
        candidate_indices=candidate_indices,
        beam_size=beam_size,
        mode="lower",
    )
    upper_limit, upper_order, upper_curve = _solve_naopc_limit_beam(
        runner=runner,
        candidate_indices=candidate_indices,
        beam_size=beam_size,
        mode="upper",
    )
    return None, lower_limit, upper_limit, lower_order, upper_order, lower_curve, upper_curve


def _solve_naopc_limits_exact(subset_drops, n_candidates):
    full_mask = (1 << n_candidates) - 1
    dp_max = np.full(full_mask + 1, -np.inf, dtype=np.float64)
    dp_min = np.full(full_mask + 1, np.inf, dtype=np.float64)
    prev_max = np.full(full_mask + 1, -1, dtype=np.int64)
    prev_min = np.full(full_mask + 1, -1, dtype=np.int64)
    dp_max[0] = 0.0
    dp_min[0] = 0.0

    for mask in range(1, full_mask + 1):
        subset_score = float(subset_drops[mask])
        bits = _mask_positions(mask, n_candidates)
        for bit in bits:
            prev_mask = mask ^ (1 << bit)
            max_candidate = dp_max[prev_mask] + subset_score
            if max_candidate > dp_max[mask]:
                dp_max[mask] = max_candidate
                prev_max[mask] = bit
            min_candidate = dp_min[prev_mask] + subset_score
            if min_candidate < dp_min[mask]:
                dp_min[mask] = min_candidate
                prev_min[mask] = bit

    upper_order = _reconstruct_order(prev_max, full_mask)
    lower_order = _reconstruct_order(prev_min, full_mask)
    upper_limit = float(dp_max[full_mask] / max(1, n_candidates))
    lower_limit = float(dp_min[full_mask] / max(1, n_candidates))
    return lower_limit, upper_limit, lower_order, upper_order


def _solve_naopc_limit_beam(*, runner, candidate_indices, beam_size, mode):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    n_candidates = int(candidate_indices.size)
    beam_size = max(1, int(beam_size))
    if mode not in {"lower", "upper"}:
        raise ValueError(f"Unsupported beam limit mode: {mode}")

    beam = [
        {
            "order": (),
            "used": frozenset(),
            "score_sum": 0.0,
            "curve": (),
        }
    ]

    for _depth in range(n_candidates):
        candidates = []
        for state in beam:
            order = state["order"]
            used = state["used"]
            score_sum = float(state["score_sum"])
            curve = state["curve"]
            for pos in range(n_candidates):
                if pos in used:
                    continue
                next_order = order + (pos,)
                selected = [int(candidate_indices[idx]) for idx in next_order]
                prefix_drop = float(runner.score_drop_for_unit_indices(selected))
                candidates.append(
                    {
                        "order": next_order,
                        "used": used | {pos},
                        "score_sum": score_sum + prefix_drop,
                        "curve": curve + (prefix_drop,),
                    }
                )
        if mode == "upper":
            candidates.sort(key=lambda item: (-item["score_sum"], item["order"]))
        else:
            candidates.sort(key=lambda item: (item["score_sum"], item["order"]))
        beam = candidates[:beam_size]

    best = beam[0]
    curve = np.asarray(best["curve"], dtype=np.float64)
    limit = float(best["score_sum"] / max(1, n_candidates))
    order = np.asarray(best["order"], dtype=np.int64)
    return limit, order, curve


def _reconstruct_order(prev, full_mask):
    order = []
    mask = int(full_mask)
    while mask:
        bit = int(prev[mask])
        if bit < 0:
            raise RuntimeError(f"Failed to reconstruct order for mask={mask}.")
        order.append(bit)
        mask ^= 1 << bit
    order.reverse()
    return np.asarray(order, dtype=np.int64)


def _curve_from_order(subset_drops, order_positions):
    mask = 0
    values = []
    for pos in np.asarray(order_positions, dtype=np.int64).tolist():
        mask |= 1 << int(pos)
        values.append(float(subset_drops[mask]))
    return np.asarray(values, dtype=np.float64)


def _curve_from_order_runner(runner, candidate_indices, order_positions):
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    selected = []
    values = []
    for pos in np.asarray(order_positions, dtype=np.int64).tolist():
        selected.append(int(candidate_indices[int(pos)]))
        values.append(float(runner.score_drop_for_unit_indices(selected)))
    return np.asarray(values, dtype=np.float64)


def _stable_descending_order(candidate_method_scores, candidate_reference, candidate_indices):
    scores = np.asarray(candidate_method_scores, dtype=np.float64).copy()
    refs = np.asarray(candidate_reference, dtype=np.float64)
    idx = np.asarray(candidate_indices, dtype=np.int64)
    scores[~np.isfinite(scores)] = -np.inf
    order = np.lexsort((idx, -refs, -scores))
    return order.astype(np.int64, copy=False)


def _normalize_unit_mode(unit_mode):
    unit_mode = str(unit_mode).strip().lower()
    if unit_mode not in {"filter", "spatial_cell"}:
        raise ValueError(f"Unsupported unit_mode: {unit_mode}. Expected 'filter' or 'spatial_cell'.")
    return unit_mode


def _normalize_limit_mode(limit_mode):
    limit_mode = str(limit_mode).strip().lower()
    if limit_mode not in {"exact", "beam"}:
        raise ValueError(f"Unsupported limit_mode: {limit_mode}. Expected 'exact' or 'beam'.")
    return limit_mode


def _normalize_beam_size(beam_size, limit_mode):
    if limit_mode != "beam":
        return None
    if beam_size is None:
        return int(DEFAULT_BEAM_SIZE)
    beam_size = int(beam_size)
    if beam_size < 1:
        raise ValueError(f"beam_size must be >= 1, got {beam_size}.")
    return beam_size


def _normalize_target_spec(target_spec):
    normalized = {}
    for key, value in dict(target_spec).items():
        if isinstance(value, (np.integer, int)):
            normalized[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            normalized[key] = float(value)
        elif isinstance(value, (list, tuple)):
            normalized[key] = [
                int(item) if isinstance(item, (np.integer, int)) else float(item) if isinstance(item, (np.floating, float)) else item
                for item in value
            ]
        else:
            normalized[key] = value
    return normalized


def _limit_mode_label(limit_mode, beam_size):
    limit_mode = _normalize_limit_mode(limit_mode)
    if limit_mode == "beam":
        return f"limit=beam(B={int(beam_size)})"
    return "limit=exact"


def _unit_scores_from_payload(payload, unit_mode):
    cond_tensor = payload["cond_tensor"]
    return _unit_scores_from_parts(_to_part_tuple(cond_tensor), unit_mode=unit_mode)


def _unit_scores_from_parts(parts, *, unit_mode):
    outputs = []
    for part in parts:
        outputs.append(_unit_scores_from_tensor(part, unit_mode=unit_mode))
    if not outputs:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(outputs, axis=0)


def _unit_scores_from_tensor(tensor, *, unit_mode):
    unit_mode = _normalize_unit_mode(unit_mode)
    arr = tensor[0].detach().cpu().numpy().astype(np.float64, copy=False)
    if unit_mode == "filter":
        if arr.ndim == 1:
            return arr
        return arr.reshape(arr.shape[0], -1).sum(axis=1)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1).sum(axis=0)


def _unit_reference_scores_from_parts(parts, *, unit_mode):
    outputs = []
    for part in parts:
        outputs.append(_unit_reference_scores_from_tensor(part, unit_mode=unit_mode))
    if not outputs:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(outputs, axis=0)


def _unit_reference_scores_from_tensor(tensor, *, unit_mode):
    unit_mode = _normalize_unit_mode(unit_mode)
    arr = tensor[0].detach().cpu().numpy().astype(np.float64, copy=False)
    if unit_mode == "filter":
        if arr.ndim == 1:
            return np.abs(arr)
        return np.abs(arr).reshape(arr.shape[0], -1).sum(axis=1)
    if arr.ndim == 1:
        return np.abs(arr)
    return np.abs(arr).reshape(arr.shape[0], -1).sum(axis=0)


def _to_part_tuple(obj):
    if torch.is_tensor(obj):
        return (obj,)
    if isinstance(obj, (list, tuple)):
        tensors = tuple(item for item in obj if torch.is_tensor(item))
        if tensors:
            return tensors
    raise TypeError(f"Expected tensor/list/tuple, got {type(obj).__name__}")


def _part_offsets(part_channels):
    offsets = []
    total = 0
    for size in part_channels:
        offsets.append(total)
        total += int(size)
    return offsets


def _split_flat_unit_indices(unit_indices, part_unit_counts):
    result = [[] for _ in part_unit_counts]
    offsets = _part_offsets(part_unit_counts)
    for flat_idx in unit_indices:
        flat_idx = int(flat_idx)
        for part_idx, offset in enumerate(offsets):
            stop = offset + int(part_unit_counts[part_idx])
            if offset <= flat_idx < stop:
                result[part_idx].append(flat_idx - offset)
                break
    return result


def _part_unit_counts(parts, *, unit_mode):
    counts = []
    for part in parts:
        per_sample = part[0]
        if unit_mode == "filter":
            counts.append(int(per_sample.shape[0]))
        elif per_sample.ndim == 1:
            counts.append(int(per_sample.shape[0]))
        else:
            counts.append(int(np.prod(per_sample.shape[1:], dtype=np.int64)))
    return counts


def _build_single_tensor_unit_labels(layer_name, tensor_shape, *, unit_mode):
    fake = torch.zeros(tensor_shape)
    return _build_multi_part_unit_labels((layer_name,), (fake,), unit_mode=unit_mode)


def _build_multi_part_unit_labels(layer_names, parts, *, unit_mode):
    labels = []
    for part_idx, part in enumerate(parts):
        layer_name = layer_names[part_idx] if part_idx < len(layer_names) else f"part_{part_idx}"
        per_sample = part[0]
        if unit_mode == "filter" or per_sample.ndim == 1:
            for local_idx in range(int(per_sample.shape[0])):
                labels.append(f"{layer_name}:f{local_idx}")
            continue
        spatial_shape = tuple(int(dim) for dim in per_sample.shape[1:])
        for flat_idx in range(int(np.prod(spatial_shape, dtype=np.int64))):
            coords = np.unravel_index(flat_idx, spatial_shape)
            labels.append(f"{layer_name}:cell{coords}")
    return labels


def _patch_single_tensor_units(out, baseline, index_tensor, *, unit_mode):
    if not torch.is_tensor(out):
        raise TypeError(f"Expected tensor output for patch hook, got {type(out).__name__}")
    patched = out.clone()
    if unit_mode == "filter" or patched.ndim <= 2:
        patched[:, index_tensor, ...] = baseline[:, index_tensor, ...]
        return patched
    flat_patched = patched.reshape(patched.shape[0], patched.shape[1], -1)
    flat_baseline = baseline.reshape(baseline.shape[0], baseline.shape[1], -1)
    flat_patched[:, :, index_tensor] = flat_baseline[:, :, index_tensor]
    return flat_patched.reshape_as(patched)


def _normalize_method_specs(task, method_specs):
    normalized = []
    seen_ids = set()
    for raw_spec in method_specs:
        spec = dict(raw_spec)
        spec["kind"] = str(spec["kind"])
        spec["name"] = str(spec.get("name") or _default_method_name(spec))
        spec["id"] = _method_id(spec)
        if spec["id"] in seen_ids:
            raise ValueError(f"Duplicate method spec id: {spec['id']}. Please use unique names/params.")
        seen_ids.add(spec["id"])
        if task == "detector" and spec["kind"] == "naa" and raw_spec.get("mode") == "fixed_query":
            raise ValueError("Detector NAA does not support mode='fixed_query'.")
        normalized.append(spec)
    return normalized


def _default_method_name(spec):
    kind = str(spec["kind"])
    if kind != "cheap_ig":
        return kind.upper() if kind in {"ig", "naa"} else kind
    seg_start = spec.get("segment_start", 0.0)
    seg_end = spec.get("segment_end", 0.1)
    selection_mode = spec.get("selection_mode", "signed")
    selection_top_k = spec.get("selection_top_k", 5000)
    return f"cheap-ig[{seg_start:g},{seg_end:g}]/{selection_mode}/k{int(selection_top_k)}"


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


def _current_task_device_label(task):
    if task == "classifier":
        return current_device_label(getattr(IG, "DEVICE", None))
    return current_device_label(getattr(IG_det, "DEVICE", None))


def _clear_all_backend_caches():
    for module in (IG, NAA, IG_det, NAA_det):
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


def _mask_positions(mask, n_candidates):
    return [idx for idx in range(int(n_candidates)) if (mask >> idx) & 1]


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
    import json

    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True)
