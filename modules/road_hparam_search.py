from __future__ import annotations

"""Staged hyperparameter search for Cheap-IG with ROAD MoRF evaluation."""

import csv
import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules.road_benchmark import benchmark_classifier_road, classifier_method_spec


DEFAULT_OUTPUT_DIR = "output/road_hparam_search_oxford_pets_100_pred_top1"
DEFAULT_CACHE_ROOT = "output/road_cache"
DEFAULT_SEARCH_FRACTION = 0.7
DEFAULT_SPLIT_SEED = 0
DEFAULT_BASELINE_N_STEPS = 128
DEFAULT_EARLY_PERCENTILES = (20, 40, 60, 80)
DEFAULT_FINAL_PERCENTILES = (10, 20, 30, 40, 50, 60, 70, 80, 90)
DEFAULT_STAGE_A_IMAGES = 20
DEFAULT_STAGE_A_LIMIT = 6
DEFAULT_STAGE_B_LIMIT = 8
DEFAULT_FINALIST_COUNT = 3
DEFAULT_STAGE_A_RHO = 0.8
DEFAULT_EPS = 1e-12


def default_cheap_ig_search_space():
    return {
        "selection_mode": "positive",
        "selection_top_k_values": [4000, 6000, 8000, 10000, 16000, 32000],
        "segment_start": 0.0,
        "segment_end_values": [0.1, 0.12, 0.15, 0.2],
        "fill_modes": ["zero", "naa_scaled"],
        "fill_rho_values": [0.6, 0.8, 1.0, 1.2],
        "n_steps_values": [24, 48, 96, 192],
        "stage_a_n_steps": 48,
        "stage_b_n_steps": 48,
        "baseline_n_steps": DEFAULT_BASELINE_N_STEPS,
        "stage_a_percentiles": list(DEFAULT_EARLY_PERCENTILES),
        "stage_b_percentiles": list(DEFAULT_EARLY_PERCENTILES),
        "stage_c_percentiles": list(DEFAULT_FINAL_PERCENTILES),
        "holdout_percentiles": list(DEFAULT_FINAL_PERCENTILES),
        "stage_a_image_count": DEFAULT_STAGE_A_IMAGES,
        "stage_a_limit": DEFAULT_STAGE_A_LIMIT,
        "stage_b_limit": DEFAULT_STAGE_B_LIMIT,
        "stage_a_hybrid_rho": DEFAULT_STAGE_A_RHO,
    }


def split_search_holdout(image_paths, search_fraction=DEFAULT_SEARCH_FRACTION, seed=DEFAULT_SPLIT_SEED):
    normalized = [str(Path(path)) for path in image_paths]
    if len(normalized) < 2:
        raise ValueError("Need at least 2 images to create search/holdout split.")
    fraction = float(search_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"search_fraction must be in (0, 1), got {fraction}.")
    rng = np.random.default_rng(int(seed))
    shuffled = list(normalized)
    rng.shuffle(shuffled)
    n_search = int(round(len(shuffled) * fraction))
    n_search = min(max(n_search, 1), len(shuffled) - 1)
    return {
        "all_image_paths": normalized,
        "shuffled_image_paths": shuffled,
        "search_image_paths": shuffled[:n_search],
        "holdout_image_paths": shuffled[n_search:],
        "search_fraction": fraction,
        "seed": int(seed),
    }


def run_staged_road_search(
    *,
    image_paths,
    layer_name,
    search_space=None,
    search_fraction=DEFAULT_SEARCH_FRACTION,
    seed=DEFAULT_SPLIT_SEED,
    cache_root=DEFAULT_CACHE_ROOT,
    noise=0.01,
    noise_seed=0,
    top_n=0,
    fd_eps=1e-3,
    clear_every=8,
    refresh_core=False,
    refresh_methods=False,
    refresh_evaluations=False,
    verbose=False,
):
    search_space = _normalize_search_space(search_space or default_cheap_ig_search_space())
    split = split_search_holdout(image_paths, search_fraction=search_fraction, seed=seed)

    search_images = split["search_image_paths"]
    holdout_images = split["holdout_image_paths"]
    stage_ab_images = search_images[: min(int(search_space["stage_a_image_count"]), len(search_images))]

    stage_a_configs = _build_stage_a_configs(search_space)
    stage_a = _evaluate_search_stage(
        stage_name="A",
        split_name="search",
        image_paths=stage_ab_images,
        candidate_configs=stage_a_configs,
        n_steps_values=[int(search_space["stage_a_n_steps"])],
        percentiles=search_space["stage_a_percentiles"],
        layer_name=layer_name,
        cache_root=cache_root,
        noise=noise,
        noise_seed=noise_seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
        refresh_evaluations=refresh_evaluations,
        verbose=verbose,
    )
    _apply_pareto_flags(stage_a["candidate_records"])
    stage_a["selected_base_keys"] = _select_stage_a_shortlist(
        stage_a["candidate_records"],
        limit=int(search_space["stage_a_limit"]),
    )
    _mark_selected_records(stage_a["candidate_records"], stage_a["selected_base_keys"])

    stage_b = _run_stage_b(
        stage_a=stage_a,
        search_space=search_space,
        image_paths=stage_ab_images,
        layer_name=layer_name,
        cache_root=cache_root,
        noise=noise,
        noise_seed=noise_seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
        refresh_evaluations=refresh_evaluations,
        verbose=verbose,
    )

    stage_c = _run_stage_c(
        stage_b=stage_b,
        search_space=search_space,
        image_paths=search_images,
        layer_name=layer_name,
        cache_root=cache_root,
        noise=noise,
        noise_seed=noise_seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
        refresh_evaluations=refresh_evaluations,
        verbose=verbose,
    )

    holdout = _run_holdout_stage(
        finalists=stage_c["finalists"],
        search_space=search_space,
        image_paths=holdout_images,
        layer_name=layer_name,
        cache_root=cache_root,
        noise=noise,
        noise_seed=noise_seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
        refresh_evaluations=refresh_evaluations,
        verbose=verbose,
    )

    candidate_rows = stage_a["candidate_records"] + stage_b["candidate_records"] + stage_c["candidate_records"]
    return {
        "task": "classifier",
        "layer_name": str(layer_name),
        "cache_root": str(cache_root),
        "search_space": search_space,
        "split": split,
        "stages": {
            "A": stage_a,
            "B": stage_b,
            "C": stage_c,
            "holdout": holdout,
        },
        "candidate_rows": candidate_rows,
        "finalists": stage_c["finalists"],
    }


def render_road_search_report(search_result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    render_tag = str(int(time.time_ns()))

    stage_a = search_result["stages"]["A"]
    stage_b = search_result["stages"]["B"]
    stage_c = search_result["stages"]["C"]
    holdout = search_result["stages"]["holdout"]

    figures = {}
    figures["stage_a_heatmap"] = _save_figure(
        _plot_stage_a_heatmap(stage_a["candidate_records"]),
        figure_dir / f"stage_a_heatmap_{render_tag}.png",
    )
    figures["stage_b_rho_heatmap"] = _save_figure(
        _plot_stage_b_rho_heatmap(stage_b["candidate_records"]),
        figure_dir / f"stage_b_rho_heatmap_{render_tag}.png",
    )
    figures["nsteps_tradeoff"] = _save_figure(
        _plot_nsteps_tradeoff(stage_c["candidate_records"]),
        figure_dir / f"nsteps_tradeoff_{render_tag}.png",
    )
    figures["search_pareto_scatter"] = _save_figure(
        _plot_search_pareto_scatter(stage_c["candidate_records"], stage_c["finalists"]),
        figure_dir / f"search_pareto_scatter_{render_tag}.png",
    )
    figures["search_parameter_effects"] = _save_figure(
        _plot_search_parameter_effects(stage_c["candidate_records"]),
        figure_dir / f"search_parameter_effects_{render_tag}.png",
    )
    figures["holdout_curves"] = _save_figure(
        _plot_holdout_curves(holdout["rows"], holdout["percentiles"], holdout["display_specs"]),
        figure_dir / f"holdout_curves_{render_tag}.png",
    )
    figures["holdout_summary"] = _save_figure(
        _plot_holdout_summary(holdout["method_records"], holdout["display_specs"]),
        figure_dir / f"holdout_summary_{render_tag}.png",
    )

    search_candidates_csv = output_dir / "search_candidates.csv"
    _write_search_candidates_csv(search_candidates_csv, search_result["candidate_rows"], holdout["method_records"])

    search_report_path = output_dir / "search_report.md"
    holdout_report_path = output_dir / "holdout_report.md"
    search_summary_path = output_dir / "search_summary.json"
    holdout_summary_path = output_dir / "holdout_summary.json"

    search_report_path.write_text(
        _build_search_report_markdown(search_result, figures, search_candidates_csv),
        encoding="utf-8",
    )
    holdout_report_path.write_text(
        _build_holdout_report_markdown(search_result, figures),
        encoding="utf-8",
    )
    search_summary_path.write_text(
        _pretty_json(
            {
                "task": search_result["task"],
                "layer_name": search_result["layer_name"],
                "cache_root": search_result["cache_root"],
                "search_space": search_result["search_space"],
                "split": search_result["split"],
                "finalists": search_result["finalists"],
                "stages": {
                    "A": _stage_json_payload(stage_a),
                    "B": _stage_json_payload(stage_b),
                    "C": _stage_json_payload(stage_c),
                },
                "figures": figures,
                "search_candidates_csv": str(search_candidates_csv),
                "search_report_path": str(search_report_path),
                "holdout_report_path": str(holdout_report_path),
            }
        ),
        encoding="utf-8",
    )
    holdout_summary_path.write_text(
        _pretty_json(
            {
                "task": search_result["task"],
                "layer_name": search_result["layer_name"],
                "split": {
                    "n_holdout_images": len(search_result["split"]["holdout_image_paths"]),
                    "holdout_image_paths": search_result["split"]["holdout_image_paths"],
                },
                "baseline_n_steps": int(search_result["search_space"]["baseline_n_steps"]),
                "finalists": search_result["finalists"],
                "method_records": holdout["method_records"],
                "pairwise_win_rate": holdout["pairwise_win_rate"],
                "figures": {
                    "holdout_curves": figures["holdout_curves"],
                    "holdout_summary": figures["holdout_summary"],
                },
            }
        ),
        encoding="utf-8",
    )

    return {
        "output_dir": str(output_dir),
        "figure_dir": str(figure_dir),
        "figures": figures,
        "search_report_path": str(search_report_path),
        "holdout_report_path": str(holdout_report_path),
        "search_summary_path": str(search_summary_path),
        "holdout_summary_path": str(holdout_summary_path),
        "search_candidates_csv": str(search_candidates_csv),
        "finalists": search_result["finalists"],
    }


def _normalize_search_space(search_space):
    normalized = dict(search_space)
    normalized["selection_mode"] = str(normalized.get("selection_mode", "positive"))
    normalized["segment_start"] = float(normalized.get("segment_start", 0.0))
    normalized["selection_top_k_values"] = [int(v) for v in normalized["selection_top_k_values"]]
    normalized["segment_end_values"] = [float(v) for v in normalized["segment_end_values"]]
    normalized["fill_modes"] = [str(v) for v in normalized["fill_modes"]]
    normalized["fill_rho_values"] = [float(v) for v in normalized["fill_rho_values"]]
    normalized["n_steps_values"] = [int(v) for v in normalized["n_steps_values"]]
    normalized["stage_a_n_steps"] = int(normalized["stage_a_n_steps"])
    normalized["stage_b_n_steps"] = int(normalized["stage_b_n_steps"])
    normalized["baseline_n_steps"] = int(normalized.get("baseline_n_steps", DEFAULT_BASELINE_N_STEPS))
    normalized["stage_a_percentiles"] = [int(v) for v in normalized["stage_a_percentiles"]]
    normalized["stage_b_percentiles"] = [int(v) for v in normalized["stage_b_percentiles"]]
    normalized["stage_c_percentiles"] = [int(v) for v in normalized["stage_c_percentiles"]]
    normalized["holdout_percentiles"] = [int(v) for v in normalized["holdout_percentiles"]]
    normalized["stage_a_image_count"] = int(normalized["stage_a_image_count"])
    normalized["stage_a_limit"] = int(normalized["stage_a_limit"])
    normalized["stage_b_limit"] = int(normalized["stage_b_limit"])
    normalized["stage_a_hybrid_rho"] = float(normalized["stage_a_hybrid_rho"])
    return normalized


def _build_stage_a_configs(search_space):
    configs = []
    for segment_end in search_space["segment_end_values"]:
        for top_k in search_space["selection_top_k_values"]:
            configs.append(
                _candidate_config(
                    selection_mode=search_space["selection_mode"],
                    selection_top_k=top_k,
                    segment_start=search_space["segment_start"],
                    segment_end=segment_end,
                    fill_mode="zero",
                    fill_rho=None,
                )
            )
            configs.append(
                _candidate_config(
                    selection_mode=search_space["selection_mode"],
                    selection_top_k=top_k,
                    segment_start=search_space["segment_start"],
                    segment_end=segment_end,
                    fill_mode="naa_scaled",
                    fill_rho=search_space["stage_a_hybrid_rho"],
                )
            )
    return configs


def _run_stage_b(
    *,
    stage_a,
    search_space,
    image_paths,
    layer_name,
    cache_root,
    noise,
    noise_seed,
    top_n,
    fd_eps,
    clear_every,
    refresh_core,
    refresh_methods,
    refresh_evaluations,
    verbose,
):
    selected_records = [record for record in stage_a["candidate_records"] if record.get("selected")]
    stage_a_by_key = {record["config_key"]: record for record in stage_a["candidate_records"]}

    zero_configs = []
    new_hybrid_configs = []
    for record in selected_records:
        if record["fill_mode"] == "zero":
            zero_configs.append(_record_to_config(record))
            continue
        for rho in search_space["fill_rho_values"]:
            candidate = _candidate_config(
                selection_mode=record["selection_mode"],
                selection_top_k=record["selection_top_k"],
                segment_start=record["segment_start"],
                segment_end=record["segment_end"],
                fill_mode="naa_scaled",
                fill_rho=rho,
            )
            if _config_key(candidate) in stage_a_by_key:
                continue
            new_hybrid_configs.append(candidate)

    stage_b = {
        "stage": "B",
        "title": "rho sweep",
        "split": "search",
        "image_paths": list(image_paths),
        "n_images": len(image_paths),
        "percentiles": list(search_space["stage_b_percentiles"]),
        "n_steps": int(search_space["stage_b_n_steps"]),
        "candidate_records": [],
    }

    for config in zero_configs:
        source = _clone_record_for_stage(stage_a_by_key[_config_key(config)], stage="B")
        source["selection_reason"] = "carried_zero"
        stage_b["candidate_records"].append(source)

    for record in selected_records:
        if record["fill_mode"] == "naa_scaled" and abs(float(record["fill_rho"]) - float(search_space["stage_a_hybrid_rho"])) <= DEFAULT_EPS:
            source = _clone_record_for_stage(record, stage="B")
            source["selection_reason"] = "carried_rho0.8"
            stage_b["candidate_records"].append(source)

    if new_hybrid_configs:
        evaluated = _evaluate_search_stage(
            stage_name="B",
            split_name="search",
            image_paths=image_paths,
            candidate_configs=new_hybrid_configs,
            n_steps_values=[int(search_space["stage_b_n_steps"])],
            percentiles=search_space["stage_b_percentiles"],
            layer_name=layer_name,
            cache_root=cache_root,
            noise=noise,
            noise_seed=noise_seed,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            refresh_core=refresh_core,
            refresh_methods=refresh_methods,
            refresh_evaluations=refresh_evaluations,
            verbose=verbose,
        )
        stage_b["candidate_records"].extend(evaluated["candidate_records"])

    _apply_pareto_flags(stage_b["candidate_records"])
    stage_b["selected_base_keys"] = _select_stage_b_shortlist(
        stage_b["candidate_records"],
        limit=int(search_space["stage_b_limit"]),
    )
    _mark_selected_records(stage_b["candidate_records"], stage_b["selected_base_keys"])
    return stage_b


def _run_stage_c(
    *,
    stage_b,
    search_space,
    image_paths,
    layer_name,
    cache_root,
    noise,
    noise_seed,
    top_n,
    fd_eps,
    clear_every,
    refresh_core,
    refresh_methods,
    refresh_evaluations,
    verbose,
):
    selected_configs = [_record_to_config(record) for record in stage_b["candidate_records"] if record.get("selected")]
    stage_c = _evaluate_search_stage(
        stage_name="C",
        split_name="search",
        image_paths=image_paths,
        candidate_configs=selected_configs,
        n_steps_values=search_space["n_steps_values"],
        percentiles=search_space["stage_c_percentiles"],
        layer_name=layer_name,
        cache_root=cache_root,
        noise=noise,
        noise_seed=noise_seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
        refresh_evaluations=refresh_evaluations,
        verbose=verbose,
    )
    _apply_pareto_flags(stage_c["candidate_records"])
    stage_c["finalists"] = _select_stage_c_finalists(stage_c["candidate_records"])
    finalist_keys = {entry["config_step_key"] for entry in stage_c["finalists"]}
    for record in stage_c["candidate_records"]:
        record["selected"] = record["config_step_key"] in finalist_keys
        record["stage_status"] = "finalist" if record["selected"] else ("pareto" if record["pareto"] else "pruned")
        labels = [
            finalist["label"]
            for finalist in stage_c["finalists"]
            if finalist["config_step_key"] == record["config_step_key"]
        ]
        record["final_label"] = "+".join(labels) if labels else None
    return stage_c


def _run_holdout_stage(
    *,
    finalists,
    search_space,
    image_paths,
    layer_name,
    cache_root,
    noise,
    noise_seed,
    top_n,
    fd_eps,
    clear_every,
    refresh_core,
    refresh_methods,
    refresh_evaluations,
    verbose,
):
    if not finalists:
        raise ValueError("No finalists available for holdout evaluation.")

    baseline_n_steps = int(search_space["baseline_n_steps"])
    percentiles = list(search_space["holdout_percentiles"])
    grouped_specs = {}

    baseline_specs = [
        classifier_method_spec("ig", name="IG"),
        classifier_method_spec("naa", name="NAA"),
    ]
    grouped_specs.setdefault(baseline_n_steps, []).extend(baseline_specs)

    finalist_display = []
    for finalist in finalists:
        config = dict(finalist["config"])
        spec = _candidate_method_spec(config)
        grouped_specs.setdefault(int(finalist["n_steps"]), []).append(spec)
        finalist_display.append(
            {
                "label": finalist["label"],
                "display_name": _display_name_with_label(spec["name"], finalist["label"]),
                "method_name": spec["name"],
                "config_step_key": finalist["config_step_key"],
                "n_steps": int(finalist["n_steps"]),
            }
        )

    grouped_results = []
    merged_rows = []
    core_rows = None
    for n_steps in sorted(grouped_specs):
        result = benchmark_classifier_road(
            image_paths=image_paths,
            method_specs=grouped_specs[n_steps],
            layer_name=layer_name,
            n_steps=n_steps,
            percentiles=percentiles,
            noise=noise,
            noise_seed=noise_seed,
            cache_root=cache_root,
            save_output=False,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            refresh_core=refresh_core,
            refresh_methods=refresh_methods,
            refresh_evaluations=refresh_evaluations,
            verbose=verbose,
        )
        grouped_results.append({"n_steps": int(n_steps), "result": result})
        merged_rows.extend(result["rows"])
        if core_rows is None:
            core_rows = result["core_rows"]

    method_specs = baseline_specs + [_candidate_method_spec(finalist["config"]) for finalist in finalists]
    method_records = _build_holdout_method_records(
        rows=merged_rows,
        method_specs=method_specs,
        finalists=finalists,
    )
    display_specs = _build_holdout_display_specs(method_specs, finalists)
    return {
        "stage": "holdout",
        "title": "holdout",
        "split": "holdout",
        "image_paths": list(image_paths),
        "n_images": len(image_paths),
        "percentiles": percentiles,
        "baseline_n_steps": baseline_n_steps,
        "rows": merged_rows,
        "core_rows": core_rows or [],
        "grouped_results": grouped_results,
        "method_records": method_records,
        "pairwise_win_rate": _pairwise_win_rate_from_rows(merged_rows, method_specs),
        "display_specs": display_specs,
        "finalists": finalists,
        "finalist_display": finalist_display,
    }


def _evaluate_search_stage(
    *,
    stage_name,
    split_name,
    image_paths,
    candidate_configs,
    n_steps_values,
    percentiles,
    layer_name,
    cache_root,
    noise,
    noise_seed,
    top_n,
    fd_eps,
    clear_every,
    refresh_core,
    refresh_methods,
    refresh_evaluations,
    verbose,
):
    candidate_records = []
    benchmark_summaries = []
    for n_steps in n_steps_values:
        method_specs = [_candidate_method_spec(config) for config in candidate_configs]
        result = benchmark_classifier_road(
            image_paths=image_paths,
            method_specs=method_specs,
            layer_name=layer_name,
            n_steps=n_steps,
            percentiles=percentiles,
            noise=noise,
            noise_seed=noise_seed,
            cache_root=cache_root,
            save_output=False,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            refresh_core=refresh_core,
            refresh_methods=refresh_methods,
            refresh_evaluations=refresh_evaluations,
            verbose=verbose,
        )
        benchmark_summaries.append({"n_steps": int(n_steps), "summary": result["summary"]})
        for method_spec in method_specs:
            stats = result["summary"]["method_summaries"][method_spec["name"]]
            candidate_records.append(
                _candidate_record(
                    stage=stage_name,
                    split_name=split_name,
                    method_spec=method_spec,
                    config=_spec_to_config(method_spec),
                    n_steps=n_steps,
                    n_images=len(image_paths),
                    percentiles=percentiles,
                    stats=stats,
                )
            )
    return {
        "stage": stage_name,
        "title": _stage_title(stage_name),
        "split": split_name,
        "image_paths": list(image_paths),
        "n_images": len(image_paths),
        "percentiles": [int(v) for v in percentiles],
        "n_steps_values": [int(v) for v in n_steps_values],
        "candidate_records": candidate_records,
        "benchmark_summaries": benchmark_summaries,
    }


def _candidate_config(*, selection_mode, selection_top_k, segment_start, segment_end, fill_mode, fill_rho):
    config = {
        "selection_mode": str(selection_mode),
        "selection_top_k": int(selection_top_k),
        "segment_start": float(segment_start),
        "segment_end": float(segment_end),
        "fill_mode": str(fill_mode),
    }
    if str(fill_mode) == "naa_scaled":
        config["fill_rho"] = float(fill_rho)
    else:
        config["fill_rho"] = None
    return config


def _candidate_method_spec(config):
    kwargs = {
        "selection_mode": config["selection_mode"],
        "selection_top_k": int(config["selection_top_k"]),
        "segment_start": float(config["segment_start"]),
        "segment_end": float(config["segment_end"]),
        "fill_mode": config["fill_mode"],
    }
    if config["fill_mode"] == "naa_scaled" and config["fill_rho"] is not None:
        kwargs["fill_rho"] = float(config["fill_rho"])
    return classifier_method_spec("cheap_ig", **kwargs)


def _record_to_config(record):
    return _candidate_config(
        selection_mode=record["selection_mode"],
        selection_top_k=record["selection_top_k"],
        segment_start=record["segment_start"],
        segment_end=record["segment_end"],
        fill_mode=record["fill_mode"],
        fill_rho=record["fill_rho"],
    )


def _spec_to_config(method_spec):
    return _candidate_config(
        selection_mode=method_spec.get("selection_mode", "positive"),
        selection_top_k=method_spec.get("selection_top_k", 0),
        segment_start=method_spec.get("segment_start", 0.0),
        segment_end=method_spec.get("segment_end", 0.1),
        fill_mode=method_spec.get("fill_mode", "zero"),
        fill_rho=method_spec.get("fill_rho"),
    )


def _candidate_record(*, stage, split_name, method_spec, config, n_steps, n_images, percentiles, stats):
    config_key = _config_key(config)
    config_step_key = _config_step_key(config, n_steps)
    fill_branch = "zero" if config["fill_mode"] == "zero" else "hybrid"
    record = {
        "stage": stage,
        "stage_title": _stage_title(stage),
        "split": split_name,
        "method_name": method_spec["name"],
        "method_id": method_spec.get("id"),
        "config": config,
        "config_key": config_key,
        "config_step_key": config_step_key,
        "selection_mode": config["selection_mode"],
        "selection_top_k": int(config["selection_top_k"]),
        "segment_start": float(config["segment_start"]),
        "segment_end": float(config["segment_end"]),
        "fill_mode": config["fill_mode"],
        "fill_branch": fill_branch,
        "fill_rho": None if config["fill_rho"] is None else float(config["fill_rho"]),
        "n_steps": int(n_steps),
        "segment_step_count": int(_segment_sample_count(config["segment_start"], config["segment_end"], n_steps)),
        "n_images": int(n_images),
        "percentiles": [int(v) for v in percentiles],
        "aoc_mean": _stats_mean(stats.get("target_logit_drop_aoc")),
        "aoc_std": _stats_std(stats.get("target_logit_drop_aoc")),
        "aoc_norm_mean": _stats_mean(stats.get("target_logit_drop_aoc_normalized")),
        "runtime_mean": _stats_mean(stats.get("runtime_s")),
        "runtime_std": _stats_std(stats.get("runtime_s")),
        "benchmark_runtime_mean": _stats_mean(stats.get("benchmark_runtime_s")),
        "benchmark_runtime_std": _stats_std(stats.get("benchmark_runtime_s")),
        "consistency_mean": _stats_mean(stats.get("road_morf_mean_consistency")),
        "consistency_std": _stats_std(stats.get("road_morf_mean_consistency")),
        "abs_error_mean": _stats_mean(stats.get("abs_error")),
        "abs_error_std": _stats_std(stats.get("abs_error")),
        "rank_mean": _stats_mean(stats.get("rank")),
        "selected": False,
        "pareto": False,
        "stage_status": "pending",
        "selection_reason": None,
        "final_label": None,
        "delta_vs_ig": None,
        "delta_vs_naa": None,
    }
    return record


def _clone_record_for_stage(record, *, stage):
    cloned = dict(record)
    cloned["stage"] = str(stage)
    cloned["stage_title"] = _stage_title(stage)
    cloned["selected"] = False
    cloned["pareto"] = False
    cloned["stage_status"] = "pending"
    cloned["selection_reason"] = None
    cloned["final_label"] = None
    return cloned


def _apply_pareto_flags(records):
    frontier = set(_pareto_keys(records))
    for record in records:
        record["pareto"] = record["config_step_key"] in frontier
        if record["stage_status"] == "pending":
            record["stage_status"] = "pareto" if record["pareto"] else "pruned"


def _select_stage_a_shortlist(records, *, limit):
    if not records:
        return []
    best_aoc = max(record["aoc_mean"] for record in records if _is_finite(record["aoc_mean"]))
    pareto_records = _sorted_candidate_records([record for record in records if record.get("pareto")])
    best_by_segment = []
    for segment_end in sorted({record["segment_end"] for record in records}):
        candidates = [record for record in records if abs(record["segment_end"] - segment_end) <= DEFAULT_EPS]
        if candidates:
            best_by_segment.append(max(candidates, key=_candidate_rank_tuple))
    near_best = [record for record in records if record["aoc_mean"] >= 0.99 * best_aoc]
    fastest_near_best = None
    if near_best:
        fastest_near_best = min(near_best, key=lambda record: (record["runtime_mean"], -record["aoc_mean"], record["method_name"]))

    ordered = []
    reason_map = {}
    for record in pareto_records:
        ordered.append(record)
        reason_map.setdefault(record["config_key"], []).append("pareto")
    for record in best_by_segment:
        ordered.append(record)
        reason_map.setdefault(record["config_key"], []).append(f"best_segment_{record['segment_end']:g}")
    if fastest_near_best is not None:
        ordered.append(fastest_near_best)
        reason_map.setdefault(fastest_near_best["config_key"], []).append("fastest_within_1pct")

    selected = []
    seen = set()
    for record in ordered:
        if record["config_key"] in seen:
            continue
        seen.add(record["config_key"])
        selected.append(record["config_key"])
        if len(selected) >= int(limit):
            break
    for record in records:
        if record["config_key"] in reason_map:
            record["selection_reason"] = ";".join(reason_map[record["config_key"]])
    return selected


def _select_stage_b_shortlist(records, *, limit):
    pareto_records = _sorted_candidate_records([record for record in records if record.get("pareto")])
    ranked = _sorted_candidate_records(records)
    selected = []
    for record in pareto_records + ranked:
        if record["config_key"] in selected:
            continue
        selected.append(record["config_key"])
        if len(selected) >= int(limit):
            break
    return selected


def _mark_selected_records(records, selected_base_keys):
    selected_base_keys = set(selected_base_keys)
    for record in records:
        record["selected"] = record["config_key"] in selected_base_keys
        record["stage_status"] = "selected" if record["selected"] else ("pareto" if record["pareto"] else "pruned")


def _select_stage_c_finalists(records):
    pareto_records = _sorted_candidate_records([record for record in records if record.get("pareto")])
    if not pareto_records:
        raise ValueError("Stage C produced no Pareto candidates.")

    best_quality = max(pareto_records, key=lambda record: (record["aoc_mean"], -record["runtime_mean"], record["method_name"]))
    fastest_pareto = min(pareto_records, key=lambda record: (record["runtime_mean"], -record["aoc_mean"], record["method_name"]))
    balanced_order = _balanced_order(pareto_records)

    finalists = []
    used = set()

    def add(label, record):
        key = record["config_step_key"]
        if key not in used:
            finalists.append(
                {
                    "label": label,
                    "method_name": record["method_name"],
                    "config": record["config"],
                    "config_key": record["config_key"],
                    "config_step_key": record["config_step_key"],
                    "n_steps": int(record["n_steps"]),
                    "segment_step_count": int(record["segment_step_count"]),
                    "aoc_mean": float(record["aoc_mean"]),
                    "runtime_mean": float(record["runtime_mean"]),
                    "fill_mode": record["fill_mode"],
                    "fill_rho": record["fill_rho"],
                    "segment_end": float(record["segment_end"]),
                    "selection_top_k": int(record["selection_top_k"]),
                }
            )
            used.add(key)
        else:
            for finalist in finalists:
                if finalist["config_step_key"] == key:
                    finalist["label"] = finalist["label"] + "+" + label
                    break

    add("best_quality", best_quality)
    add("fastest_pareto", fastest_pareto)
    if balanced_order:
        add("best_balanced", balanced_order[0])

    extra_index = 1
    for record in balanced_order[1:]:
        if len(finalists) >= DEFAULT_FINALIST_COUNT:
            break
        if record["config_step_key"] in used:
            continue
        add(f"pareto_extra_{extra_index}", record)
        extra_index += 1

    return finalists[:DEFAULT_FINALIST_COUNT]


def _balanced_order(records):
    if not records:
        return []
    aocs = np.asarray([record["aoc_mean"] for record in records], dtype=np.float64)
    runtimes = np.asarray([record["runtime_mean"] for record in records], dtype=np.float64)
    aoc_min = float(np.nanmin(aocs))
    aoc_max = float(np.nanmax(aocs))
    runtime_min = float(np.nanmin(runtimes))
    runtime_max = float(np.nanmax(runtimes))

    ordered = []
    for record in records:
        aoc_norm = 0.0 if abs(aoc_max - aoc_min) <= DEFAULT_EPS else (record["aoc_mean"] - aoc_min) / (aoc_max - aoc_min)
        runtime_norm = (
            0.0
            if abs(runtime_max - runtime_min) <= DEFAULT_EPS
            else (record["runtime_mean"] - runtime_min) / (runtime_max - runtime_min)
        )
        distance = float(np.sqrt((1.0 - aoc_norm) ** 2 + runtime_norm**2))
        ordered.append((distance, -record["aoc_mean"], record["runtime_mean"], record["method_name"], record))
    ordered.sort(key=lambda item: item[:4])
    return [item[-1] for item in ordered]


def _build_holdout_method_records(*, rows, method_specs, finalists):
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method_name"], []).append(row)
    finalist_label_map = {finalist["method_name"]: finalist["label"] for finalist in finalists}

    method_records = []
    ig_mean = None
    naa_mean = None
    for spec in method_specs:
        method_rows = by_method.get(spec["name"], [])
        record = {
            "method_name": spec["name"],
            "label": finalist_label_map.get(spec["name"]),
            "kind": spec["kind"],
            "selection_top_k": spec.get("selection_top_k"),
            "segment_end": spec.get("segment_end"),
            "fill_mode": spec.get("fill_mode"),
            "fill_rho": spec.get("fill_rho"),
            "aoc": _stats_record([row.get("target_logit_drop_aoc") for row in method_rows]),
            "aoc_normalized": _stats_record([row.get("target_logit_drop_aoc_normalized") for row in method_rows]),
            "runtime": _stats_record([row.get("method_duration_s") for row in method_rows]),
            "benchmark_runtime": _stats_record([row.get("benchmark_duration_s") for row in method_rows]),
            "consistency": _stats_record([row.get("road_morf_mean_consistency") for row in method_rows]),
            "abs_error": _stats_record([row.get("abs_error") for row in method_rows]),
        }
        if spec["name"] == "IG":
            ig_mean = record["aoc"]["mean"]
        if spec["name"] == "NAA":
            naa_mean = record["aoc"]["mean"]
        method_records.append(record)

    for record in method_records:
        record["delta_vs_ig"] = (
            None if ig_mean is None or not _is_finite(record["aoc"]["mean"]) else float(record["aoc"]["mean"] - ig_mean)
        )
        record["delta_vs_naa"] = (
            None if naa_mean is None or not _is_finite(record["aoc"]["mean"]) else float(record["aoc"]["mean"] - naa_mean)
        )
    return method_records


def _build_holdout_display_specs(method_specs, finalists):
    label_map = {finalist["method_name"]: finalist["label"] for finalist in finalists}
    display_specs = []
    for spec in method_specs:
        display_specs.append(
            {
                "name": spec["name"],
                "display_name": _display_name_with_label(spec["name"], label_map.get(spec["name"])),
                "kind": spec["kind"],
            }
        )
    return display_specs


def _pairwise_win_rate_from_rows(rows, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image_path"], {})[row["method_name"]] = row.get("target_logit_drop_aoc")
    pairwise = {left: {} for left in method_names}
    for left in method_names:
        for right in method_names:
            if left == right:
                pairwise[left][right] = float("nan")
                continue
            wins = []
            for image_scores in by_image.values():
                left_score = image_scores.get(left)
                right_score = image_scores.get(right)
                if not _is_finite(left_score) or not _is_finite(right_score):
                    continue
                wins.append(1.0 if left_score > right_score else 0.0)
            pairwise[left][right] = float(np.mean(wins)) if wins else float("nan")
    return pairwise


def _plot_stage_a_heatmap(records):
    zero_records = [record for record in records if record["fill_mode"] == "zero"]
    hybrid_records = [
        record
        for record in records
        if record["fill_mode"] == "naa_scaled" and abs(float(record["fill_rho"]) - DEFAULT_STAGE_A_RHO) <= DEFAULT_EPS
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _draw_candidate_heatmap(
        axes[0],
        zero_records,
        title="Stage A: zero tail",
        x_values=sorted({record["selection_top_k"] for record in records}),
        y_values=sorted({record["segment_end"] for record in records}),
        x_getter=lambda record: record["selection_top_k"],
        y_getter=lambda record: record["segment_end"],
    )
    _draw_candidate_heatmap(
        axes[1],
        hybrid_records,
        title="Stage A: hybrid rho=0.8",
        x_values=sorted({record["selection_top_k"] for record in records}),
        y_values=sorted({record["segment_end"] for record in records}),
        x_getter=lambda record: record["selection_top_k"],
        y_getter=lambda record: record["segment_end"],
    )
    fig.suptitle("Stage A topology screen")
    return fig


def _plot_stage_b_rho_heatmap(records):
    segment_values = sorted({record["segment_end"] for record in records if record["fill_mode"] == "naa_scaled"})
    if not segment_values:
        return _placeholder_figure("Stage B rho sweep", "No hybrid records available.")
    fig, axes = plt.subplots(1, len(segment_values), figsize=(6 * len(segment_values), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, segment_end in zip(axes, segment_values):
        segment_records = [
            record
            for record in records
            if record["fill_mode"] == "naa_scaled" and abs(record["segment_end"] - segment_end) <= DEFAULT_EPS
        ]
        _draw_candidate_heatmap(
            axis,
            segment_records,
            title=f"Stage B: segment_end={segment_end:g}",
            x_values=sorted({record["fill_rho"] for record in segment_records}),
            y_values=sorted({record["selection_top_k"] for record in segment_records}),
            x_getter=lambda record: record["fill_rho"],
            y_getter=lambda record: record["selection_top_k"],
            x_label="rho",
            y_label="top_k",
        )
    fig.suptitle("Stage B rho sweep")
    return fig


def _plot_nsteps_tradeoff(records):
    if not records:
        return _placeholder_figure("n_steps refinement", "No stage C records.")
    grouped = {}
    for record in records:
        grouped.setdefault(record["config_key"], []).append(record)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(grouped), 1)))
    for color, (config_key, config_records) in zip(colors, sorted(grouped.items())):
        config_records = sorted(config_records, key=lambda record: record["n_steps"])
        label = _short_config_label(config_records[0], include_n_steps=False)
        axes[0].plot(
            [record["n_steps"] for record in config_records],
            [record["aoc_mean"] for record in config_records],
            marker="o",
            color=color,
            label=label,
        )
        axes[1].plot(
            [record["n_steps"] for record in config_records],
            [record["runtime_mean"] for record in config_records],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].set_title("AOC vs n_steps")
    axes[1].set_title("Runtime vs n_steps")
    for axis in axes:
        axis.set_xlabel("n_steps")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Target logit drop AOC")
    axes[1].set_ylabel("Mean runtime (s)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Stage C n_steps refinement")
    return fig


def _plot_search_pareto_scatter(records, finalists):
    if not records:
        return _placeholder_figure("Search Pareto frontier", "No stage C records.")
    sorted_records = sorted(records, key=lambda record: (record["runtime_mean"], -record["aoc_mean"], record["method_name"]))
    top_k_values = sorted({record["selection_top_k"] for record in sorted_records})
    top_k_palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
    top_k_colors = {top_k: top_k_palette[idx % len(top_k_palette)] for idx, top_k in enumerate(top_k_values)}
    segment_values = sorted({record["segment_end"] for record in sorted_records})
    segment_markers_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    segment_markers = {
        segment_end: segment_markers_cycle[idx % len(segment_markers_cycle)]
        for idx, segment_end in enumerate(segment_values)
    }
    fill_values = sorted(
        {
            "zero" if record["fill_mode"] == "zero" else f"rho={float(record['fill_rho']):g}"
            for record in sorted_records
        }
    )
    fill_palette = ["#4d4d4d", "#1b7837", "#2166ac", "#762a83", "#b2182b", "#8c510a"]
    fill_edge_colors = {fill_key: fill_palette[idx % len(fill_palette)] for idx, fill_key in enumerate(fill_values)}
    n_steps_values = sorted({record["n_steps"] for record in sorted_records})
    n_steps_palette = ["#003f5c", "#2f4b7c", "#006d77", "#7b2cbf", "#9a031e", "#5f0f40"]
    n_steps_text_colors = {
        n_steps: n_steps_palette[idx % len(n_steps_palette)] for idx, n_steps in enumerate(n_steps_values)
    }
    finalists_map = {entry["config_step_key"]: entry["label"] for entry in finalists}
    n_cols = 2 if len(sorted_records) <= 16 else 3
    rows_per_col = int(np.ceil(len(sorted_records) / max(n_cols, 1)))
    fig_height = max(7.5, 5.8 + 0.28 * rows_per_col)
    fig = plt.figure(figsize=(12.5, fig_height), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.9, 1.7],
        height_ratios=[3.0, max(1.2, 0.55 + 0.22 * rows_per_col)],
    )
    ax = fig.add_subplot(grid[0, 0])
    side_ax = fig.add_subplot(grid[0, 1])
    legend_ax = fig.add_subplot(grid[1, :])
    side_ax.axis("off")
    legend_ax.axis("off")

    numbered_entries = []
    fig.canvas.draw()
    used_label_positions = []
    offset_cycle = [(7, 7), (7, -10), (-12, 7), (-12, -10), (0, 13), (0, -14), (14, 0), (-14, 0)]
    for point_idx, record in enumerate(sorted_records, start=1):
        marker = segment_markers.get(record["segment_end"], "o")
        fill_key = "zero" if record["fill_mode"] == "zero" else f"rho={float(record['fill_rho']):g}"
        face_color = top_k_colors.get(record["selection_top_k"], "#1f77b4")
        edge_color = fill_edge_colors.get(fill_key, "#4d4d4d")
        point_size = 110 if record["pareto"] else 60
        ax.scatter(
            record["runtime_mean"],
            record["aoc_mean"],
            marker=marker,
            facecolor=face_color,
            edgecolor=edge_color,
            s=point_size,
            alpha=0.92 if record["pareto"] else 0.75,
            linewidth=1.6,
        )
        if record["selected"]:
            ax.scatter(
                record["runtime_mean"],
                record["aoc_mean"],
                marker=marker,
                facecolors="none",
                edgecolors="black",
                s=point_size + 38,
                linewidths=1.5,
            )
        data_xy = np.asarray(ax.transData.transform((record["runtime_mean"], record["aoc_mean"])), dtype=np.float64)
        chosen_offset = offset_cycle[0]
        chosen_anchor = data_xy + np.asarray(chosen_offset, dtype=np.float64) * fig.dpi / 72.0
        for offset in offset_cycle:
            candidate_anchor = data_xy + np.asarray(offset, dtype=np.float64) * fig.dpi / 72.0
            if all(np.linalg.norm(candidate_anchor - prev_anchor) >= 18.0 for prev_anchor in used_label_positions):
                chosen_offset = offset
                chosen_anchor = candidate_anchor
                break
        used_label_positions.append(chosen_anchor)
        ax.annotate(
            str(point_idx),
            (record["runtime_mean"], record["aoc_mean"]),
            textcoords="offset points",
            xytext=chosen_offset,
            fontsize=8,
            fontweight="bold",
            color=n_steps_text_colors.get(record["n_steps"], "black"),
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
        )
        label_prefix = finalists_map.get(record["config_step_key"])
        record_label = dict(record)
        record_label["final_label"] = None
        if label_prefix:
            entry_label = f"{point_idx}. [{_compact_final_label(label_prefix)}] {_compact_config_label(record_label, include_n_steps=True)}"
        else:
            entry_label = f"{point_idx}. {_compact_config_label(record, include_n_steps=True)}"
        numbered_entries.append(entry_label)

    ax.set_title("Stage C Pareto frontier")
    ax.set_xlabel("Mean runtime (s)")
    ax.set_ylabel("Target logit drop AOC")
    ax.grid(alpha=0.25)

    side_ax.set_xlim(0, 1)
    side_ax.set_ylim(0, 1)
    y = 0.98
    side_ax.text(0.02, y, "Top-k (fill color)", ha="left", va="top", fontsize=10, fontweight="bold")
    y -= 0.08
    for top_k in top_k_values:
        side_ax.scatter(0.08, y, s=85, marker="o", facecolor=top_k_colors[top_k], edgecolor="black", linewidth=0.8)
        side_ax.text(0.16, y, _format_top_k_short(top_k), va="center", ha="left", fontsize=9)
        y -= 0.065

    y -= 0.02
    side_ax.text(0.02, y, "segment_end (shape)", ha="left", va="top", fontsize=10, fontweight="bold")
    y -= 0.08
    for segment_end in segment_values:
        side_ax.scatter(0.08, y, s=85, marker=segment_markers[segment_end], facecolor="#bbbbbb", edgecolor="black", linewidth=0.8)
        side_ax.text(0.16, y, f"s{segment_end:g}", va="center", ha="left", fontsize=9)
        y -= 0.065

    y -= 0.02
    side_ax.text(0.02, y, "fill/rho (edge)", ha="left", va="top", fontsize=10, fontweight="bold")
    y -= 0.08
    for fill_key in fill_values:
        side_ax.scatter(0.08, y, s=85, marker="o", facecolor="white", edgecolor=fill_edge_colors[fill_key], linewidth=2.0)
        side_ax.text(0.16, y, "z" if fill_key == "zero" else fill_key.replace("rho=", "h"), va="center", ha="left", fontsize=9)
        y -= 0.065

    y -= 0.02
    side_ax.text(0.02, y, "n_steps (label color)", ha="left", va="top", fontsize=10, fontweight="bold")
    y -= 0.08
    for n_steps in n_steps_values:
        side_ax.text(0.08, y, f"n{n_steps}", va="center", ha="left", fontsize=9, color=n_steps_text_colors[n_steps], fontweight="bold")
        y -= 0.06

    y -= 0.02
    side_ax.text(0.02, y, "Other", ha="left", va="top", fontsize=10, fontweight="bold")
    y -= 0.08
    side_ax.scatter(0.08, y, s=60, marker="o", facecolor="#bbbbbb", edgecolor="#666666", linewidth=1.4)
    side_ax.text(0.16, y, "regular", va="center", ha="left", fontsize=9)
    y -= 0.065
    side_ax.scatter(0.08, y, s=110, marker="o", facecolor="#bbbbbb", edgecolor="#666666", linewidth=1.6)
    side_ax.text(0.16, y, "Pareto", va="center", ha="left", fontsize=9)
    y -= 0.065
    side_ax.scatter(0.08, y, s=110, marker="o", facecolor="none", edgecolor="black", linewidth=1.5)
    side_ax.text(0.16, y, "selected finalist", va="center", ha="left", fontsize=9)

    legend_ax.text(
        0.01,
        0.98,
        "Point labels map to configs below",
        va="top",
        ha="left",
        fontsize=9,
        fontweight="bold",
        transform=legend_ax.transAxes,
    )
    for col_idx in range(n_cols):
        start = col_idx * rows_per_col
        end = min(start + rows_per_col, len(numbered_entries))
        if start >= end:
            continue
        legend_ax.text(
            0.01 + col_idx / n_cols,
            0.86,
            "\n".join(numbered_entries[start:end]),
            va="top",
            ha="left",
            fontsize=8,
            family="monospace",
            transform=legend_ax.transAxes,
        )
    return fig


def _plot_search_parameter_effects(records):
    if not records:
        return _placeholder_figure("Search parameter effects", "No stage C records.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = [
        ("top_k effect", sorted({record["selection_top_k"] for record in records}), lambda record: record["selection_top_k"]),
        ("segment_end effect", sorted({record["segment_end"] for record in records}), lambda record: record["segment_end"]),
        (
            "rho effect (hybrid only)",
            sorted({record["fill_rho"] for record in records if record["fill_mode"] == "naa_scaled"}),
            lambda record: record["fill_rho"],
        ),
        ("n_steps effect", sorted({record["n_steps"] for record in records}), lambda record: record["n_steps"]),
    ]
    for axis, (title, values, getter) in zip(axes.reshape(-1), plots):
        if title.startswith("rho"):
            subset = [record for record in records if record["fill_mode"] == "naa_scaled"]
        else:
            subset = list(records)
        grouped = []
        labels = []
        means = []
        for value in values:
            group = [record["aoc_mean"] for record in subset if getter(record) == value and _is_finite(record["aoc_mean"])]
            grouped.append(group if group else [float("nan")])
            labels.append(str(value))
            means.append(float(np.nanmean(grouped[-1])))
        axis.boxplot(grouped, labels=labels, widths=0.6)
        axis.plot(np.arange(1, len(means) + 1), means, marker="o", color="#d62728")
        axis.set_title(title)
        axis.set_ylabel("AOC")
        axis.grid(alpha=0.2)
    fig.suptitle("Stage C parameter effects on search split")
    return fig


def _plot_holdout_curves(rows, percentiles, display_specs):
    if not rows:
        return _placeholder_figure("Holdout curves", "No holdout rows.")
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    for spec in display_specs:
        method_rows = [row for row in rows if row["method_name"] == spec["name"]]
        if not method_rows:
            continue
        curves = np.asarray([row["target_logit_drop_curve"] for row in method_rows], dtype=np.float64)
        mean_curve = np.nanmean(curves, axis=0)
        ax.plot(percentiles, mean_curve, marker="o", linewidth=2, label=spec["display_name"])
    ax.set_title("Holdout ROAD target-logit curves")
    ax.set_xlabel("Removed pixels (%)")
    ax.set_ylabel("Target logit drop")
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return fig


def _plot_holdout_summary(method_records, display_specs):
    if not method_records:
        return _placeholder_figure("Holdout summary", "No holdout methods.")
    display_map = {spec["name"]: spec["display_name"] for spec in display_specs}
    ordered = sorted(method_records, key=lambda record: record["aoc"]["mean"])
    labels = [display_map.get(record["method_name"], record["method_name"]) for record in ordered]
    means = [record["aoc"]["mean"] for record in ordered]
    stds = [record["aoc"]["std"] for record in ordered]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    ax.errorbar(means, y, xerr=stds, fmt="o", color="#1f77b4", ecolor="black", elinewidth=1.2, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target logit drop AOC (higher is better)")
    ax.set_title("Holdout summary")
    ax.grid(axis="x", alpha=0.25)
    return fig


def _draw_candidate_heatmap(
    axis,
    records,
    *,
    title,
    x_values,
    y_values,
    x_getter,
    y_getter,
    x_label="top_k",
    y_label="segment_end",
):
    if not records:
        axis.set_title(title)
        axis.text(0.5, 0.5, "No data", ha="center", va="center")
        axis.set_axis_off()
        return

    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for record in records:
        row = y_values.index(y_getter(record))
        col = x_values.index(x_getter(record))
        matrix[row, col] = record["aoc_mean"]
    im = axis.imshow(matrix, cmap="magma")
    axis.set_title(title)
    axis.set_xticks(np.arange(len(x_values)))
    axis.set_xticklabels([str(value) for value in x_values])
    axis.set_yticks(np.arange(len(y_values)))
    axis.set_yticklabels([str(value) for value in y_values])
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if _is_finite(value):
                axis.text(col, row, f"{value:.3f}", ha="center", va="center", color="white", fontsize=9)
    plt.colorbar(im, ax=axis, shrink=0.85)


def _build_search_report_markdown(search_result, figures, search_candidates_csv):
    stage_a = search_result["stages"]["A"]
    stage_b = search_result["stages"]["B"]
    stage_c = search_result["stages"]["C"]
    finalists = search_result["finalists"]
    lines = [
        "# ROAD Hyperparameter Search",
        "",
        f"- layer_name=`{search_result['layer_name']}`",
        f"- n_images_total={len(search_result['split']['all_image_paths'])}",
        f"- search_images={len(search_result['split']['search_image_paths'])}",
        f"- holdout_images={len(search_result['split']['holdout_image_paths'])}",
        f"- split_seed={search_result['split']['seed']}",
        f"- cache_root=`{search_result['cache_root']}`",
        "",
        "## Finalists",
        "",
        _build_finalists_table(finalists),
        "",
        "## Stage A",
        "",
        _build_search_stage_table(stage_a["candidate_records"]),
        "",
        "## Stage B",
        "",
        _build_search_stage_table(stage_b["candidate_records"]),
        "",
        "## Stage C",
        "",
        _build_search_stage_table(stage_c["candidate_records"]),
        "",
        "## Figures",
        "",
        f"`search_candidates.csv`: `{search_candidates_csv}`",
        "",
    ]
    for key in (
        "stage_a_heatmap",
        "stage_b_rho_heatmap",
        "nsteps_tradeoff",
        "search_pareto_scatter",
        "search_parameter_effects",
    ):
        lines.extend([f"### {key}", "", f"![]({figures[key]})", ""])
    return "\n".join(lines).rstrip() + "\n"


def _build_holdout_report_markdown(search_result, figures):
    holdout = search_result["stages"]["holdout"]
    lines = [
        "# ROAD Hyperparameter Search Holdout",
        "",
        f"- layer_name=`{search_result['layer_name']}`",
        f"- holdout_images={holdout['n_images']}",
        f"- baseline_n_steps={holdout['baseline_n_steps']}",
        "",
        "## Finalists",
        "",
        _build_finalists_table(search_result["finalists"]),
        "",
        "## Holdout Summary",
        "",
        _build_holdout_table(holdout["method_records"], holdout["display_specs"]),
        "",
        "## Figures",
        "",
    ]
    for key in ("holdout_curves", "holdout_summary"):
        lines.extend([f"### {key}", "", f"![]({figures[key]})", ""])
    return "\n".join(lines).rstrip() + "\n"


def _build_finalists_table(finalists):
    headers = [
        "Label",
        "Method",
        "AOC",
        "Runtime (s)",
        "top_k",
        "segment_end",
        "fill",
        "seg steps",
        "n_steps",
    ]
    rows = [headers, ["---"] * len(headers)]
    for finalist in finalists:
        rows.append(
            [
                finalist["label"],
                _candidate_method_spec(finalist["config"])["name"],
                f"{finalist['aoc_mean']:.4f}",
                f"{finalist['runtime_mean']:.4f}",
                str(finalist["selection_top_k"]),
                f"{finalist['segment_end']:g}",
                _fill_display(finalist["fill_mode"], finalist["fill_rho"]),
                str(finalist["segment_step_count"]),
                str(finalist["n_steps"]),
            ]
        )
    return _markdown_table(rows)


def _build_search_stage_table(records):
    headers = [
        "Method",
        "AOC",
        "Runtime (s)",
        "Benchmark (s)",
        "delta vs IG",
        "delta vs NAA",
        "seg steps",
        "Status",
        "Reason",
    ]
    rows = [headers, ["---"] * len(headers)]
    for record in _sorted_candidate_records(records):
        rows.append(
            [
                _short_config_label(record, include_n_steps=True),
                _format_number(record["aoc_mean"]),
                _format_number(record["runtime_mean"]),
                _format_number(record["benchmark_runtime_mean"]),
                _format_number(record["delta_vs_ig"]),
                _format_number(record["delta_vs_naa"]),
                str(record["segment_step_count"]),
                record["stage_status"],
                record["selection_reason"] or "—",
            ]
        )
    return _markdown_table(rows)


def _build_holdout_table(method_records, display_specs):
    display_map = {spec["name"]: spec["display_name"] for spec in display_specs}
    headers = [
        "Method",
        "AOC",
        "Runtime (s)",
        "Benchmark (s)",
        "delta vs IG",
        "delta vs NAA",
        "Consistency",
        "Abs Error",
    ]
    rows = [headers, ["---"] * len(headers)]
    order = {spec["name"]: idx for idx, spec in enumerate(display_specs)}
    for record in sorted(method_records, key=lambda item: order.get(item["method_name"], 999)):
        rows.append(
            [
                display_map.get(record["method_name"], record["method_name"]),
                _format_stats(record["aoc"]),
                _format_stats(record["runtime"]),
                _format_stats(record["benchmark_runtime"]),
                _format_number(record["delta_vs_ig"]),
                _format_number(record["delta_vs_naa"]),
                _format_stats(record["consistency"]),
                _format_stats(record["abs_error"]),
            ]
        )
    return _markdown_table(rows)


def _write_search_candidates_csv(path, candidate_rows, holdout_records):
    holdout_lookup = {record["method_name"]: record for record in holdout_records}
    fieldnames = [
        "stage",
        "stage_title",
        "method_name",
        "config_key",
        "config_step_key",
        "selection_mode",
        "selection_top_k",
        "segment_start",
        "segment_end",
        "segment_step_count",
        "fill_mode",
        "fill_rho",
        "n_steps",
        "aoc_mean",
        "aoc_std",
        "runtime_mean",
        "runtime_std",
        "benchmark_runtime_mean",
        "consistency_mean",
        "abs_error_mean",
        "pareto",
        "selected",
        "stage_status",
        "selection_reason",
        "final_label",
        "delta_vs_ig",
        "delta_vs_naa",
        "holdout_aoc_mean",
        "holdout_runtime_mean",
        "holdout_delta_vs_ig",
        "holdout_delta_vs_naa",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_rows:
            holdout = holdout_lookup.get(row["method_name"], {})
            writer.writerow(
                {
                    **{field: row.get(field) for field in fieldnames if field in row},
                    "holdout_aoc_mean": _nested_mean(holdout.get("aoc")),
                    "holdout_runtime_mean": _nested_mean(holdout.get("runtime")),
                    "holdout_delta_vs_ig": holdout.get("delta_vs_ig"),
                    "holdout_delta_vs_naa": holdout.get("delta_vs_naa"),
                }
            )


def _stage_json_payload(stage):
    return {
        "stage": stage["stage"],
        "title": stage["title"],
        "split": stage["split"],
        "n_images": stage["n_images"],
        "image_paths": stage["image_paths"],
        "percentiles": stage.get("percentiles"),
        "n_steps": stage.get("n_steps"),
        "n_steps_values": stage.get("n_steps_values"),
        "selected_base_keys": stage.get("selected_base_keys"),
        "finalists": stage.get("finalists"),
        "candidate_records": stage["candidate_records"],
    }


def _config_key(config):
    material = json.dumps(
        {
            "selection_mode": config["selection_mode"],
            "selection_top_k": int(config["selection_top_k"]),
            "segment_start": float(config["segment_start"]),
            "segment_end": float(config["segment_end"]),
            "fill_mode": config["fill_mode"],
            "fill_rho": None if config["fill_rho"] is None else float(config["fill_rho"]),
        },
        sort_keys=True,
    )
    digest = hashlib.md5(material.encode("utf-8")).hexdigest()[:8]
    return f"cheapig_{digest}"


def _config_step_key(config, n_steps):
    return f"{_config_key(config)}_n{int(n_steps)}"


def _stage_title(stage_name):
    return {
        "A": "topology screen",
        "B": "rho sweep",
        "C": "n_steps refinement",
        "holdout": "holdout",
    }.get(stage_name, str(stage_name))


def _sorted_candidate_records(records):
    return sorted(records, key=lambda record: (-record["aoc_mean"], record["runtime_mean"], record["method_name"]))


def _candidate_rank_tuple(record):
    return (record["aoc_mean"], -record["runtime_mean"], record["method_name"])


def _pareto_keys(records):
    valid = [record for record in records if _is_finite(record["aoc_mean"]) and _is_finite(record["runtime_mean"])]
    frontier = []
    for record in valid:
        dominated = False
        for other in valid:
            if other["config_step_key"] == record["config_step_key"]:
                continue
            if (
                other["aoc_mean"] >= record["aoc_mean"] - DEFAULT_EPS
                and other["runtime_mean"] <= record["runtime_mean"] + DEFAULT_EPS
                and (
                    other["aoc_mean"] > record["aoc_mean"] + DEFAULT_EPS
                    or other["runtime_mean"] < record["runtime_mean"] - DEFAULT_EPS
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(record["config_step_key"])
    return frontier


def _display_name_with_label(method_name, label):
    if not label:
        return method_name
    return f"{label}: {method_name}"


def _short_config_label(record, *, include_n_steps):
    parts = [
        f"k={record['selection_top_k']}",
        f"seg={record['segment_end']:g}",
        _fill_display(record["fill_mode"], record["fill_rho"]),
    ]
    if include_n_steps:
        parts.append(f"n={record['n_steps']}")
    label = ", ".join(parts)
    if record.get("final_label"):
        label = f"{record['final_label']} - {label}"
    return label


def _compact_config_label(record, *, include_n_steps):
    parts = [
        _format_top_k_short(record["selection_top_k"]),
        f"s{float(record['segment_end']):g}",
        _compact_fill_display(record["fill_mode"], record["fill_rho"]),
    ]
    if include_n_steps:
        parts.append(f"n{int(record['n_steps'])}")
    return " | ".join(parts)


def _format_top_k_short(top_k):
    top_k = int(top_k)
    if top_k % 1000 == 0:
        return f"{top_k // 1000}k"
    return str(top_k)


def _compact_fill_display(fill_mode, fill_rho):
    if fill_mode == "zero":
        return "z"
    return f"h{float(fill_rho):g}"


def _compact_final_label(label):
    mapping = {
        "best_quality": "BQ",
        "best_balanced": "BB",
        "fastest_pareto": "FP",
    }
    return mapping.get(str(label), str(label))


def _fill_display(fill_mode, fill_rho):
    if fill_mode == "zero":
        return "zero"
    return f"hybrid rho={float(fill_rho):g}"


def _segment_sample_count(segment_start, segment_end, n_steps):
    n_steps = int(n_steps)
    start = float(segment_start)
    end = float(segment_end)
    count = 0
    for step_idx in range(1, n_steps + 1):
        alpha = step_idx / float(n_steps)
        if end < 1.0:
            if alpha >= start and alpha < end:
                count += 1
        else:
            if alpha >= start and alpha <= end:
                count += 1
    return count


def _save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _placeholder_figure(title, text):
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.set_title(title)
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()
    return fig


def _stats_record(values):
    arr = np.asarray([float(value) for value in values if _is_finite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def _stats_mean(stats):
    if not isinstance(stats, dict):
        return float("nan")
    return float(stats.get("mean", float("nan")))


def _stats_std(stats):
    if not isinstance(stats, dict):
        return float("nan")
    return float(stats.get("std", float("nan")))


def _nested_mean(stats):
    if not isinstance(stats, dict):
        return None
    value = stats.get("mean")
    return None if not _is_finite(value) else float(value)


def _format_number(value):
    return "—" if value is None or not _is_finite(value) else f"{float(value):.4f}"


def _format_stats(stats):
    if not isinstance(stats, dict):
        return "—"
    mean = stats.get("mean")
    std = stats.get("std")
    if not _is_finite(mean):
        return "—"
    return f"{float(mean):.4f} +- {float(std):.4f}"


def _markdown_table(rows):
    return "\n".join("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)


def _pretty_json(value):
    return json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False) + "\n"


def _is_finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False
