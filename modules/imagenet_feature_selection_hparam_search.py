from __future__ import annotations

"""Staged hyperparameter search for Cheap-IG on ImageNet feature-selection benchmark."""

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules import alpha_segment_benchmark as seg
from modules import feature_selection_benchmark as fs
from modules import imagenet_feature_selection_benchmark as ifs


DEFAULT_OUTPUT_DIR = "output/imagenet_feature_selection_hparam_search"
DEFAULT_CACHE_ROOT = ifs.DEFAULT_CACHE_ROOT
DEFAULT_SEARCH_FRACTION = 0.75
DEFAULT_SPLIT_SEED = 0
DEFAULT_BASELINE_N_STEPS = ifs.DEFAULT_N_STEPS
DEFAULT_STAGE_A_TASK_COUNT = 2
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
        "stage_a_task_count": DEFAULT_STAGE_A_TASK_COUNT,
        "stage_a_limit": DEFAULT_STAGE_A_LIMIT,
        "stage_b_limit": DEFAULT_STAGE_B_LIMIT,
        "stage_a_hybrid_rho": DEFAULT_STAGE_A_RHO,
        "k_values": list(ifs.DEFAULT_K_VALUES),
    }


def split_search_holdout_tasks(class_tasks, search_fraction=DEFAULT_SEARCH_FRACTION, seed=DEFAULT_SPLIT_SEED):
    tasks = [dict(task) for task in class_tasks]
    if len(tasks) < 2:
        raise ValueError("Need at least 2 tasks to create search/holdout split.")
    fraction = float(search_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"search_fraction must be in (0, 1), got {fraction}.")
    rng = np.random.default_rng(int(seed))
    order = list(range(len(tasks)))
    rng.shuffle(order)
    shuffled = [tasks[idx] for idx in order]
    n_search = int(round(len(shuffled) * fraction))
    n_search = min(max(n_search, 1), len(shuffled) - 1)
    search_tasks = shuffled[:n_search]
    holdout_tasks = shuffled[n_search:]
    return {
        "all_tasks": tasks,
        "shuffled_task_names": [task["name"] for task in shuffled],
        "search_tasks": search_tasks,
        "holdout_tasks": holdout_tasks,
        "search_fraction": fraction,
        "seed": int(seed),
    }


def run_staged_imagenet_feature_selection_search(
    *,
    image_paths,
    imagenet_root=ifs.DEFAULT_IMAGENET_VAL_ROOT,
    class_tasks=None,
    layer_name=ifs.DEFAULT_LAYER_NAME,
    search_space=None,
    search_fraction=DEFAULT_SEARCH_FRACTION,
    seed=DEFAULT_SPLIT_SEED,
    train_per_class=ifs.DEFAULT_TRAIN_PER_CLASS,
    eval_per_class=ifs.DEFAULT_EVAL_PER_CLASS,
    channel_aggregation=ifs.DEFAULT_CHANNEL_AGGREGATION,
    selection_rule=ifs.DEFAULT_SELECTION_RULE,
    cache_root=DEFAULT_CACHE_ROOT,
    top_n=0,
    fd_eps=1e-3,
    clear_every=8,
    refresh_core=False,
    refresh_methods=False,
):
    search_space = _normalize_search_space(search_space or default_cheap_ig_search_space())
    if class_tasks is None:
        class_tasks = ifs.default_imagenet_feature_selection_tasks(imagenet_root, random_seed=seed)
    split = split_search_holdout_tasks(class_tasks, search_fraction=search_fraction, seed=seed)
    search_tasks = split["search_tasks"]
    holdout_tasks = split["holdout_tasks"]
    stage_ab_tasks = search_tasks[: min(int(search_space["stage_a_task_count"]), len(search_tasks))]

    stage_a_configs = _build_stage_a_configs(search_space)
    stage_a = _evaluate_search_stage(
        stage_name="A",
        split_name="search",
        image_paths=image_paths,
        class_tasks=stage_ab_tasks,
        candidate_configs=stage_a_configs,
        n_steps_values=[int(search_space["stage_a_n_steps"])],
        k_values=search_space["k_values"],
        layer_name=layer_name,
        train_per_class=train_per_class,
        eval_per_class=eval_per_class,
        channel_aggregation=channel_aggregation,
        selection_rule=selection_rule,
        cache_root=cache_root,
        seed=seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
    )
    _apply_pareto_flags(stage_a["candidate_records"])
    stage_a["selected_base_keys"] = _select_stage_a_shortlist(stage_a["candidate_records"], limit=int(search_space["stage_a_limit"]))
    _mark_selected_records(stage_a["candidate_records"], stage_a["selected_base_keys"])

    stage_b = _run_stage_b(
        stage_a=stage_a,
        search_space=search_space,
        image_paths=image_paths,
        class_tasks=stage_ab_tasks,
        layer_name=layer_name,
        train_per_class=train_per_class,
        eval_per_class=eval_per_class,
        channel_aggregation=channel_aggregation,
        selection_rule=selection_rule,
        cache_root=cache_root,
        seed=seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
    )

    stage_c = _run_stage_c(
        stage_b=stage_b,
        search_space=search_space,
        image_paths=image_paths,
        class_tasks=search_tasks,
        layer_name=layer_name,
        train_per_class=train_per_class,
        eval_per_class=eval_per_class,
        channel_aggregation=channel_aggregation,
        selection_rule=selection_rule,
        cache_root=cache_root,
        seed=seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
    )

    holdout = _run_holdout_stage(
        finalists=stage_c["finalists"],
        search_space=search_space,
        image_paths=image_paths,
        class_tasks=holdout_tasks,
        layer_name=layer_name,
        train_per_class=train_per_class,
        eval_per_class=eval_per_class,
        channel_aggregation=channel_aggregation,
        selection_rule=selection_rule,
        cache_root=cache_root,
        seed=seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
    )

    candidate_rows = stage_a["candidate_records"] + stage_b["candidate_records"] + stage_c["candidate_records"]
    return {
        "task": "classifier",
        "dataset": "imagenet_val",
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


def render_imagenet_feature_selection_search_report(search_result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    render_tag = str(int(time.time_ns()))

    stage_a = search_result["stages"]["A"]
    stage_b = search_result["stages"]["B"]
    stage_c = search_result["stages"]["C"]
    holdout = search_result["stages"]["holdout"]

    figures = {
        "stage_a_heatmap": seg._save_figure(
            _plot_stage_a_heatmap(stage_a["candidate_records"]),
            figure_dir / f"stage_a_heatmap_{render_tag}.png",
        ),
        "stage_b_rho_heatmap": seg._save_figure(
            _plot_stage_b_rho_heatmap(stage_b["candidate_records"]),
            figure_dir / f"stage_b_rho_heatmap_{render_tag}.png",
        ),
        "nsteps_tradeoff": seg._save_figure(
            _plot_nsteps_tradeoff(stage_c["candidate_records"]),
            figure_dir / f"nsteps_tradeoff_{render_tag}.png",
        ),
        "search_pareto_scatter": seg._save_figure(
            _plot_search_pareto_scatter(stage_c["candidate_records"], stage_c["finalists"]),
            figure_dir / f"search_pareto_scatter_{render_tag}.png",
        ),
        "search_parameter_effects": seg._save_figure(
            _plot_search_parameter_effects(stage_c["candidate_records"]),
            figure_dir / f"search_parameter_effects_{render_tag}.png",
        ),
        "holdout_curves": seg._save_figure(
            _plot_holdout_curves(holdout["rows"], holdout["display_specs"]),
            figure_dir / f"holdout_curves_{render_tag}.png",
        ),
        "holdout_summary": seg._save_figure(
            _plot_holdout_summary(holdout["method_records"]),
            figure_dir / f"holdout_summary_{render_tag}.png",
        ),
    }

    search_candidates_csv = output_dir / "search_candidates.csv"
    _write_search_candidates_csv(search_candidates_csv, search_result["candidate_rows"])

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
                "dataset": search_result["dataset"],
                "layer_name": search_result["layer_name"],
                "cache_root": search_result["cache_root"],
                "search_space": search_result["search_space"],
                "split": {
                    "shuffled_task_names": search_result["split"]["shuffled_task_names"],
                    "search_task_names": [task["name"] for task in search_result["split"]["search_tasks"]],
                    "holdout_task_names": [task["name"] for task in search_result["split"]["holdout_tasks"]],
                    "search_fraction": search_result["split"]["search_fraction"],
                    "seed": search_result["split"]["seed"],
                },
                "finalists": search_result["finalists"],
                "stages": {
                    "A": _stage_json_payload(search_result["stages"]["A"]),
                    "B": _stage_json_payload(search_result["stages"]["B"]),
                    "C": _stage_json_payload(search_result["stages"]["C"]),
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
                "dataset": search_result["dataset"],
                "layer_name": search_result["layer_name"],
                "holdout_task_names": [task["name"] for task in search_result["split"]["holdout_tasks"]],
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
    normalized["selection_top_k_values"] = [int(v) for v in normalized["selection_top_k_values"]]
    normalized["segment_start"] = float(normalized.get("segment_start", 0.0))
    normalized["segment_end_values"] = [float(v) for v in normalized["segment_end_values"]]
    normalized["fill_modes"] = [str(v) for v in normalized["fill_modes"]]
    normalized["fill_rho_values"] = [float(v) for v in normalized["fill_rho_values"]]
    normalized["n_steps_values"] = [int(v) for v in normalized["n_steps_values"]]
    normalized["stage_a_n_steps"] = int(normalized["stage_a_n_steps"])
    normalized["stage_b_n_steps"] = int(normalized["stage_b_n_steps"])
    normalized["baseline_n_steps"] = int(normalized.get("baseline_n_steps", DEFAULT_BASELINE_N_STEPS))
    normalized["stage_a_task_count"] = int(normalized.get("stage_a_task_count", DEFAULT_STAGE_A_TASK_COUNT))
    normalized["stage_a_limit"] = int(normalized["stage_a_limit"])
    normalized["stage_b_limit"] = int(normalized["stage_b_limit"])
    normalized["stage_a_hybrid_rho"] = float(normalized["stage_a_hybrid_rho"])
    normalized["k_values"] = [int(v) for v in normalized.get("k_values", ifs.DEFAULT_K_VALUES)]
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
    class_tasks,
    layer_name,
    train_per_class,
    eval_per_class,
    channel_aggregation,
    selection_rule,
    cache_root,
    seed,
    top_n,
    fd_eps,
    clear_every,
    refresh_core,
    refresh_methods,
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
        "task_names": [task["name"] for task in class_tasks],
        "n_tasks": len(class_tasks),
        "k_values": list(search_space["k_values"]),
        "n_steps": int(search_space["stage_b_n_steps"]),
        "candidate_records": [],
    }

    for config in zero_configs:
        source = _clone_record_for_stage(stage_a_by_key[_config_key(config)], stage="B")
        source["n_steps"] = int(search_space["stage_b_n_steps"])
        source["segment_step_count"] = int(_segment_sample_count(source["segment_start"], source["segment_end"], source["n_steps"]))
        source["config_step_key"] = _config_step_key(source["config"], source["n_steps"])
        source["selection_reason"] = "carried_zero"
        stage_b["candidate_records"].append(source)

    for record in selected_records:
        if record["fill_mode"] == "naa_scaled" and abs(float(record["fill_rho"]) - float(search_space["stage_a_hybrid_rho"])) <= DEFAULT_EPS:
            source = _clone_record_for_stage(record, stage="B")
            source["n_steps"] = int(search_space["stage_b_n_steps"])
            source["segment_step_count"] = int(_segment_sample_count(source["segment_start"], source["segment_end"], source["n_steps"]))
            source["config_step_key"] = _config_step_key(source["config"], source["n_steps"])
            source["selection_reason"] = "carried_rho0.8"
            stage_b["candidate_records"].append(source)

    if new_hybrid_configs:
        evaluated = _evaluate_search_stage(
            stage_name="B",
            split_name="search",
            image_paths=image_paths,
            class_tasks=class_tasks,
            candidate_configs=new_hybrid_configs,
            n_steps_values=[int(search_space["stage_b_n_steps"])],
            k_values=search_space["k_values"],
            layer_name=layer_name,
            train_per_class=train_per_class,
            eval_per_class=eval_per_class,
            channel_aggregation=channel_aggregation,
            selection_rule=selection_rule,
            cache_root=cache_root,
            seed=seed,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            refresh_core=refresh_core,
            refresh_methods=refresh_methods,
        )
        stage_b["candidate_records"].extend(evaluated["candidate_records"])

    _apply_pareto_flags(stage_b["candidate_records"])
    stage_b["selected_base_keys"] = _select_stage_b_shortlist(stage_b["candidate_records"], limit=int(search_space["stage_b_limit"]))
    _mark_selected_records(stage_b["candidate_records"], stage_b["selected_base_keys"])
    return stage_b


def _run_stage_c(
    *,
    stage_b,
    search_space,
    image_paths,
    class_tasks,
    layer_name,
    train_per_class,
    eval_per_class,
    channel_aggregation,
    selection_rule,
    cache_root,
    seed,
    top_n,
    fd_eps,
    clear_every,
    refresh_core,
    refresh_methods,
):
    selected_configs = [_record_to_config(record) for record in stage_b["candidate_records"] if record.get("selected")]
    stage_c = _evaluate_search_stage(
        stage_name="C",
        split_name="search",
        image_paths=image_paths,
        class_tasks=class_tasks,
        candidate_configs=selected_configs,
        n_steps_values=search_space["n_steps_values"],
        k_values=search_space["k_values"],
        layer_name=layer_name,
        train_per_class=train_per_class,
        eval_per_class=eval_per_class,
        channel_aggregation=channel_aggregation,
        selection_rule=selection_rule,
        cache_root=cache_root,
        seed=seed,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
    )
    _apply_pareto_flags(stage_c["candidate_records"])
    stage_c["finalists"] = _select_stage_c_finalists(stage_c["candidate_records"])
    finalist_keys = {entry["config_step_key"] for entry in stage_c["finalists"]}
    for record in stage_c["candidate_records"]:
        record["selected"] = record["config_step_key"] in finalist_keys
        record["stage_status"] = "finalist" if record["selected"] else ("pareto" if record["pareto"] else "pruned")
        labels = [finalist["label"] for finalist in stage_c["finalists"] if finalist["config_step_key"] == record["config_step_key"]]
        record["final_label"] = "+".join(labels) if labels else None
    return stage_c


def _run_holdout_stage(
    *,
    finalists,
    search_space,
    image_paths,
    class_tasks,
    layer_name,
    train_per_class,
    eval_per_class,
    channel_aggregation,
    selection_rule,
    cache_root,
    seed,
    top_n,
    fd_eps,
    clear_every,
    refresh_core,
    refresh_methods,
):
    if not finalists:
        raise ValueError("No finalists available for holdout evaluation.")
    baseline_n_steps = int(search_space["baseline_n_steps"])

    grouped_specs = {}
    baseline_specs = [
        ifs.classifier_method_spec("ig", name="IG", segment_start=0.0, segment_end=1.0),
        ifs.classifier_method_spec("naa", name="NAA", segment_start=0.0, segment_end=1.0),
    ]
    grouped_specs.setdefault(baseline_n_steps, []).extend(baseline_specs)

    finalist_display = []
    for finalist in finalists:
        spec = _candidate_method_spec(finalist["config"])
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
    merged_method_rows = []
    merged_summaries = []
    for n_steps in sorted(grouped_specs):
        result = ifs.benchmark_imagenet_feature_selection(
            image_paths=image_paths,
            method_specs=grouped_specs[n_steps],
            class_tasks=class_tasks,
            layer_name=layer_name,
            n_steps=n_steps,
            train_per_class=train_per_class,
            eval_per_class=eval_per_class,
            k_values=search_space["k_values"],
            channel_aggregation=channel_aggregation,
            selection_rule=selection_rule,
            random_seed=seed,
            cache_root=cache_root,
            save_output=False,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            refresh_core=refresh_core,
            refresh_methods=refresh_methods,
        )
        grouped_results.append({"n_steps": int(n_steps), "result": result})
        merged_rows.extend(result["rows"])
        merged_method_rows.extend(result["method_rows"])
        merged_summaries.append({"n_steps": int(n_steps), "summary": result["summary"]})

    method_specs = baseline_specs + [_candidate_method_spec(finalist["config"]) for finalist in finalists]
    method_records = _build_holdout_method_records(
        rows=merged_rows,
        method_rows=merged_method_rows,
        method_specs=method_specs,
        finalists=finalists,
        k_values=search_space["k_values"],
    )
    return {
        "stage": "holdout",
        "title": "holdout",
        "split": "holdout",
        "task_names": [task["name"] for task in class_tasks],
        "n_tasks": len(class_tasks),
        "k_values": list(search_space["k_values"]),
        "rows": merged_rows,
        "method_rows": merged_method_rows,
        "grouped_results": grouped_results,
        "summary_rows": _merge_holdout_summary_rows(merged_summaries),
        "method_records": method_records,
        "pairwise_win_rate": _pairwise_win_rate_from_rows(merged_rows, method_specs),
        "display_specs": _build_holdout_display_specs(method_specs, finalists),
        "finalists": finalists,
        "finalist_display": finalist_display,
    }


def _evaluate_search_stage(
    *,
    stage_name,
    split_name,
    image_paths,
    class_tasks,
    candidate_configs,
    n_steps_values,
    k_values,
    layer_name,
    train_per_class,
    eval_per_class,
    channel_aggregation,
    selection_rule,
    cache_root,
    seed,
    top_n,
    fd_eps,
    clear_every,
    refresh_core,
    refresh_methods,
):
    candidate_records = []
    benchmark_summaries = []
    for n_steps in n_steps_values:
        method_specs = [_candidate_method_spec(config) for config in candidate_configs]
        result = ifs.benchmark_imagenet_feature_selection(
            image_paths=image_paths,
            method_specs=method_specs,
            class_tasks=class_tasks,
            layer_name=layer_name,
            n_steps=n_steps,
            train_per_class=train_per_class,
            eval_per_class=eval_per_class,
            k_values=k_values,
            channel_aggregation=channel_aggregation,
            selection_rule=selection_rule,
            random_seed=seed,
            cache_root=cache_root,
            save_output=False,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            refresh_core=refresh_core,
            refresh_methods=refresh_methods,
        )
        benchmark_summaries.append({"n_steps": int(n_steps), "summary": result["summary"]})
        for method_spec in method_specs:
            candidate_records.append(
                _candidate_record(
                    stage=stage_name,
                    split_name=split_name,
                    method_spec=method_spec,
                    config=_spec_to_config(method_spec),
                    n_steps=n_steps,
                    task_names=[task["name"] for task in class_tasks],
                    k_values=k_values,
                    result=result,
                )
            )
    return {
        "stage": stage_name,
        "title": _stage_title(stage_name),
        "split": split_name,
        "task_names": [task["name"] for task in class_tasks],
        "n_tasks": len(class_tasks),
        "k_values": [int(v) for v in k_values],
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
    config["fill_rho"] = float(fill_rho) if str(fill_mode) == "naa_scaled" and fill_rho is not None else None
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
    return ifs.classifier_method_spec("cheap_ig", **kwargs)


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


def _candidate_record(*, stage, split_name, method_spec, config, n_steps, task_names, k_values, result):
    config_key = _config_key(config)
    config_step_key = _config_step_key(config, n_steps)
    fill_branch = "zero" if config["fill_mode"] == "zero" else "hybrid"
    task_stats = _task_metric_stats(result["rows"], method_spec["name"])
    summary_rows = [row for row in result["summary"]["summary_rows"] if row["method_name"] == method_spec["name"]]
    runtime_stats = seg._stats_record(
        [row["method_duration_s"] for row in result["method_rows"] if row["method_name"] == method_spec["name"]]
    )
    abs_error_stats = seg._stats_record(
        [row.get("abs_error") for row in result["method_rows"] if row["method_name"] == method_spec["name"]]
    )
    global_best_row = _best_summary_row(summary_rows)
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
        "task_names": list(task_names),
        "n_tasks": len(task_names),
        "k_values": [int(v) for v in k_values],
        "accuracy_auc_mean": float(task_stats["accuracy_auc"]["mean"]),
        "accuracy_auc_std": float(task_stats["accuracy_auc"]["std"]),
        "macro_f1_auc_mean": float(task_stats["macro_f1_auc"]["mean"]),
        "macro_f1_auc_std": float(task_stats["macro_f1_auc"]["std"]),
        "best_accuracy_mean": float(task_stats["best_accuracy"]["mean"]),
        "best_accuracy_std": float(task_stats["best_accuracy"]["std"]),
        "best_macro_f1_mean": float(task_stats["best_macro_f1"]["mean"]),
        "best_macro_f1_std": float(task_stats["best_macro_f1"]["std"]),
        "best_k_global": None if global_best_row is None else int(global_best_row["k"]),
        "best_accuracy_global": None if global_best_row is None else float(global_best_row["accuracy_mean"]),
        "runtime_mean": float(runtime_stats["mean"]),
        "runtime_std": float(runtime_stats["std"]),
        "abs_error_mean": float(abs_error_stats["mean"]),
        "abs_error_std": float(abs_error_stats["std"]),
        "stage_status": "pending",
        "selection_reason": None,
        "selected": False,
        "pareto": False,
        "final_label": None,
    }
    return record


def _task_metric_stats(rows, method_name):
    by_task = {}
    for row in rows:
        if row["method_name"] != method_name:
            continue
        by_task.setdefault(row["task_name"], []).append(row)
    accuracy_auc = []
    macro_f1_auc = []
    best_accuracy = []
    best_macro_f1 = []
    for task_rows in by_task.values():
        accuracy_auc.append(float(np.mean([row["accuracy"] for row in task_rows])))
        macro_f1_auc.append(float(np.mean([row["macro_f1"] for row in task_rows])))
        best_row = max(task_rows, key=lambda row: (row["accuracy"], row["macro_f1"], -int(row["k"])))
        best_accuracy.append(float(best_row["accuracy"]))
        best_macro_f1.append(float(best_row["macro_f1"]))
    return {
        "accuracy_auc": seg._stats_record(accuracy_auc),
        "macro_f1_auc": seg._stats_record(macro_f1_auc),
        "best_accuracy": seg._stats_record(best_accuracy),
        "best_macro_f1": seg._stats_record(best_macro_f1),
    }


def _best_summary_row(summary_rows):
    if not summary_rows:
        return None
    ranked = sorted(summary_rows, key=lambda row: (-float(row["accuracy_mean"]), -float(row["macro_f1_mean"]), int(row["k"])))
    return ranked[0]


def _clone_record_for_stage(record, *, stage):
    cloned = dict(record)
    cloned["stage"] = stage
    cloned["stage_title"] = _stage_title(stage)
    cloned["stage_status"] = "pending"
    cloned["selection_reason"] = None
    cloned["selected"] = False
    cloned["pareto"] = False
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
    best_score = max(record["accuracy_auc_mean"] for record in records if _is_finite(record["accuracy_auc_mean"]))
    pareto_records = _sorted_candidate_records([record for record in records if record.get("pareto")])
    best_by_segment = []
    for segment_end in sorted({record["segment_end"] for record in records}):
        candidates = [record for record in records if abs(record["segment_end"] - segment_end) <= DEFAULT_EPS]
        if candidates:
            best_by_segment.append(max(candidates, key=_candidate_rank_tuple))
    near_best = [record for record in records if record["accuracy_auc_mean"] >= 0.99 * best_score]
    fastest_near_best = None
    if near_best:
        fastest_near_best = min(near_best, key=lambda record: (record["runtime_mean"], -record["accuracy_auc_mean"], record["method_name"]))

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

    best_quality = max(pareto_records, key=lambda record: (record["accuracy_auc_mean"], -record["runtime_mean"], record["method_name"]))
    fastest_pareto = min(pareto_records, key=lambda record: (record["runtime_mean"], -record["accuracy_auc_mean"], record["method_name"]))
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
                    "accuracy_auc_mean": float(record["accuracy_auc_mean"]),
                    "best_accuracy_mean": float(record["best_accuracy_mean"]),
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
    qualities = np.asarray([record["accuracy_auc_mean"] for record in records], dtype=np.float64)
    runtimes = np.asarray([record["runtime_mean"] for record in records], dtype=np.float64)
    quality_min = float(np.nanmin(qualities))
    quality_max = float(np.nanmax(qualities))
    runtime_min = float(np.nanmin(runtimes))
    runtime_max = float(np.nanmax(runtimes))

    ordered = []
    for record in records:
        quality_norm = 0.0 if abs(quality_max - quality_min) <= DEFAULT_EPS else (record["accuracy_auc_mean"] - quality_min) / (quality_max - quality_min)
        runtime_norm = 0.0 if abs(runtime_max - runtime_min) <= DEFAULT_EPS else (record["runtime_mean"] - runtime_min) / (runtime_max - runtime_min)
        distance = float(np.sqrt((1.0 - quality_norm) ** 2 + runtime_norm**2))
        ordered.append((distance, -record["accuracy_auc_mean"], record["runtime_mean"], record["method_name"], record))
    ordered.sort(key=lambda item: item[:4])
    return [item[-1] for item in ordered]


def _build_holdout_method_records(*, rows, method_rows, method_specs, finalists, k_values):
    finalist_label_map = {finalist["method_name"]: finalist["label"] for finalist in finalists}
    records = []
    ig_mean = None
    naa_mean = None
    for spec in method_specs:
        method_name = spec["name"]
        method_task_stats = _task_metric_stats(rows, method_name)
        runtime_stats = seg._stats_record([row["method_duration_s"] for row in method_rows if row["method_name"] == method_name])
        abs_error_stats = seg._stats_record([row.get("abs_error") for row in method_rows if row["method_name"] == method_name])
        summary_rows = [row for row in _summary_rows_from_rows(rows, [method_name], k_values) if row["method_name"] == method_name]
        best_row = _best_summary_row(summary_rows)
        record = {
            "method_name": method_name,
            "label": finalist_label_map.get(method_name),
            "kind": spec["kind"],
            "selection_top_k": spec.get("selection_top_k"),
            "segment_end": spec.get("segment_end"),
            "fill_mode": spec.get("fill_mode"),
            "fill_rho": spec.get("fill_rho"),
            "accuracy_auc": method_task_stats["accuracy_auc"],
            "macro_f1_auc": method_task_stats["macro_f1_auc"],
            "best_accuracy": method_task_stats["best_accuracy"],
            "best_macro_f1": method_task_stats["best_macro_f1"],
            "best_k_global": None if best_row is None else int(best_row["k"]),
            "best_accuracy_global": None if best_row is None else float(best_row["accuracy_mean"]),
            "runtime": runtime_stats,
            "abs_error": abs_error_stats,
        }
        if method_name == "IG":
            ig_mean = record["accuracy_auc"]["mean"]
        if method_name == "NAA":
            naa_mean = record["accuracy_auc"]["mean"]
        records.append(record)
    for record in records:
        record["delta_vs_ig"] = None if ig_mean is None or not _is_finite(record["accuracy_auc"]["mean"]) else float(record["accuracy_auc"]["mean"] - ig_mean)
        record["delta_vs_naa"] = None if naa_mean is None or not _is_finite(record["accuracy_auc"]["mean"]) else float(record["accuracy_auc"]["mean"] - naa_mean)
    return records


def _build_holdout_display_specs(method_specs, finalists):
    label_map = {finalist["method_name"]: finalist["label"] for finalist in finalists}
    return [
        {
            "name": spec["name"],
            "display_name": _display_name_with_label(spec["name"], label_map.get(spec["name"])),
            "kind": spec["kind"],
        }
        for spec in method_specs
    ]


def _merge_holdout_summary_rows(summaries):
    rows = []
    for entry in summaries:
        for row in entry["summary"]["summary_rows"]:
            cloned = dict(row)
            cloned["n_steps"] = int(entry["n_steps"])
            rows.append(cloned)
    return rows


def _summary_rows_from_rows(rows, method_names, k_values):
    summary_rows = []
    for method_name in method_names:
        for k in k_values:
            subset = [row for row in rows if row["method_name"] == method_name and int(row["k"]) == int(k)]
            summary_rows.append(
                {
                    "method_name": method_name,
                    "k": int(k),
                    "accuracy_mean": float(seg._stats_record([row["accuracy"] for row in subset])["mean"]),
                    "macro_f1_mean": float(seg._stats_record([row["macro_f1"] for row in subset])["mean"]),
                }
            )
    return summary_rows


def _pairwise_win_rate_from_rows(rows, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    by_task_k = {}
    for row in rows:
        by_task_k.setdefault((row["task_name"], int(row["k"])), {})[row["method_name"]] = row["accuracy"]
    pairwise = {left: {} for left in method_names}
    for left in method_names:
        for right in method_names:
            if left == right:
                pairwise[left][right] = float("nan")
                continue
            wins = []
            for cell_scores in by_task_k.values():
                left_score = cell_scores.get(left)
                right_score = cell_scores.get(right)
                if not _is_finite(left_score) or not _is_finite(right_score):
                    continue
                wins.append(1.0 if left_score > right_score else 0.0)
            pairwise[left][right] = float(np.mean(wins)) if wins else float("nan")
    return pairwise


def _plot_stage_a_heatmap(records):
    zero_records = [record for record in records if record["fill_mode"] == "zero"]
    hybrid_records = [record for record in records if record["fill_mode"] == "naa_scaled" and abs(float(record["fill_rho"]) - DEFAULT_STAGE_A_RHO) <= DEFAULT_EPS]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
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
    fig.suptitle("Stage A topology screen (accuracy AUC over k)")
    return fig


def _plot_stage_b_rho_heatmap(records):
    segment_values = sorted({record["segment_end"] for record in records if record["fill_mode"] == "naa_scaled"})
    if not segment_values:
        return _placeholder_figure("Stage B rho sweep", "No hybrid records available.")
    fig, axes = plt.subplots(1, len(segment_values), figsize=(6 * len(segment_values), 4.8), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, segment_end in zip(axes, segment_values):
        segment_records = [record for record in records if record["fill_mode"] == "naa_scaled" and abs(record["segment_end"] - segment_end) <= DEFAULT_EPS]
        _draw_candidate_heatmap(
            axis,
            segment_records,
            title=f"segment_end={segment_end:g}",
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
    for color, (_, config_records) in zip(colors, sorted(grouped.items())):
        config_records = sorted(config_records, key=lambda record: record["n_steps"])
        label = _short_config_label(config_records[0], include_n_steps=False)
        axes[0].plot([record["n_steps"] for record in config_records], [record["accuracy_auc_mean"] for record in config_records], marker="o", color=color, label=label)
        axes[1].plot([record["n_steps"] for record in config_records], [record["runtime_mean"] for record in config_records], marker="o", color=color, label=label)
    axes[0].set_title("Accuracy AUC vs n_steps")
    axes[1].set_title("Runtime vs n_steps")
    axes[0].set_ylabel("Mean accuracy over k")
    axes[1].set_ylabel("Mean runtime (s)")
    for axis in axes:
        axis.set_xlabel("n_steps")
        axis.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.suptitle("Stage C n_steps refinement")
    return fig


def _plot_search_pareto_scatter(records, finalists):
    if not records:
        return _placeholder_figure("Search Pareto frontier", "No stage C records.")
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    palette = plt.cm.tab10(np.linspace(0, 1, max(len({record["selection_top_k"] for record in records}), 1)))
    color_map = {top_k: palette[idx % len(palette)] for idx, top_k in enumerate(sorted({record["selection_top_k"] for record in records}))}
    marker_map = {segment_end: marker for segment_end, marker in zip(sorted({record["segment_end"] for record in records}), ["o", "s", "^", "D", "P", "X"])}
    finalist_keys = {finalist["config_step_key"]: finalist["label"] for finalist in finalists}
    for record in sorted(records, key=lambda row: (row["runtime_mean"], -row["accuracy_auc_mean"])):
        ax.scatter(
            record["runtime_mean"],
            record["accuracy_auc_mean"],
            s=120 if record["pareto"] else 70,
            marker=marker_map[record["segment_end"]],
            color=color_map[record["selection_top_k"]],
            edgecolor="black" if record["pareto"] else "white",
            linewidth=1.2,
            alpha=0.95 if record["pareto"] else 0.8,
        )
        if record["pareto"] or record["config_step_key"] in finalist_keys:
            label = finalist_keys.get(record["config_step_key"], _short_config_label(record, include_n_steps=True))
            ax.annotate(label, (record["runtime_mean"], record["accuracy_auc_mean"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Mean method runtime (s)")
    ax.set_ylabel("Mean accuracy over k")
    ax.set_title("Stage C Pareto frontier")
    ax.grid(alpha=0.25)
    return fig


def _plot_search_parameter_effects(records):
    if not records:
        return _placeholder_figure("Parameter effects", "No stage C records.")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = [
        ("selection_top_k", "top_k"),
        ("segment_end", "segment_end"),
        ("fill_branch", "fill branch"),
        ("n_steps", "n_steps"),
    ]
    for axis, (key, label) in zip(axes.flat, plots):
        values = sorted({record[key] for record in records}, key=lambda value: (str(type(value)), value))
        data = [[record["accuracy_auc_mean"] for record in records if record[key] == value] for value in values]
        axis.boxplot(data, tick_labels=[str(value) for value in values], widths=0.6)
        axis.set_title(label)
        axis.set_ylabel("Mean accuracy over k")
        axis.grid(alpha=0.2)
    fig.suptitle("Stage C parameter effects")
    return fig


def _plot_holdout_curves(rows, display_specs):
    if not rows:
        return _placeholder_figure("Holdout curves", "No holdout rows.")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for method_idx, spec in enumerate(display_specs):
        subset = [row for row in rows if row["method_name"] == spec["name"]]
        k_values = sorted({int(row["k"]) for row in subset})
        accuracy = []
        accuracy_std = []
        macro_f1 = []
        macro_f1_std = []
        for k in k_values:
            k_rows = [row for row in subset if int(row["k"]) == int(k)]
            acc_stats = seg._stats_record([row["accuracy"] for row in k_rows])
            f1_stats = seg._stats_record([row["macro_f1"] for row in k_rows])
            accuracy.append(acc_stats["mean"])
            accuracy_std.append(acc_stats["std"])
            macro_f1.append(f1_stats["mean"])
            macro_f1_std.append(f1_stats["std"])
        color = palette(method_idx % 10)
        x = np.asarray(k_values, dtype=np.int64)
        y_acc = np.asarray(accuracy, dtype=np.float64)
        y_acc_std = np.asarray(accuracy_std, dtype=np.float64)
        y_f1 = np.asarray(macro_f1, dtype=np.float64)
        y_f1_std = np.asarray(macro_f1_std, dtype=np.float64)
        axes[0].plot(x, y_acc, marker="o", linewidth=2, label=spec["display_name"], color=color)
        axes[0].fill_between(x, y_acc - y_acc_std, y_acc + y_acc_std, alpha=0.18, color=color)
        axes[1].plot(x, y_f1, marker="o", linewidth=2, label=spec["display_name"], color=color)
        axes[1].fill_between(x, y_f1 - y_f1_std, y_f1 + y_f1_std, alpha=0.18, color=color)
    axes[0].set_title("Holdout accuracy")
    axes[1].set_title("Holdout macro-F1")
    for axis in axes:
        axis.set_xlabel("Selected filters (k)")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Accuracy")
    axes[1].set_ylabel("Macro-F1")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    return fig


def _plot_holdout_summary(method_records):
    if not method_records:
        return _placeholder_figure("Holdout summary", "No holdout method records.")
    sorted_records = sorted(method_records, key=lambda row: (-row["accuracy_auc"]["mean"], -row["best_accuracy"]["mean"], row["method_name"]))
    y = np.arange(len(sorted_records))
    fig, ax = plt.subplots(figsize=(9.5, 0.65 * len(sorted_records) + 1.8), constrained_layout=True)
    means = np.asarray([record["accuracy_auc"]["mean"] for record in sorted_records], dtype=np.float64)
    stds = np.asarray([record["accuracy_auc"]["std"] for record in sorted_records], dtype=np.float64)
    ax.errorbar(means, y, xerr=stds, fmt="o", capsize=4, color="#1f77b4")
    ax.set_yticks(y)
    ax.set_yticklabels([_display_name_with_label(record["method_name"], record.get("label")) for record in sorted_records])
    ax.invert_yaxis()
    ax.set_xlabel("Mean accuracy over k")
    ax.set_title("Holdout summary")
    ax.grid(alpha=0.25, axis="x")
    for yi, record in enumerate(sorted_records):
        ax.text(means[yi] + max(stds[yi], 0.002) + 0.005, yi, f"best@k={record['best_k_global']}, acc={seg._format_number(record['best_accuracy_global'])}", va="center", fontsize=8)
    return fig


def _draw_candidate_heatmap(ax, records, *, title, x_values, y_values, x_getter, y_getter, x_label="top_k", y_label="segment_end"):
    matrix = np.full((len(y_values), len(x_values)), np.nan, dtype=np.float64)
    for record in records:
        x_value = x_getter(record)
        y_value = y_getter(record)
        row_idx = y_values.index(y_value)
        col_idx = x_values.index(x_value)
        matrix[row_idx, col_idx] = float(record["accuracy_auc_mean"])
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([str(v) for v in x_values], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{value:g}" if isinstance(value, float) else str(value) for value in y_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text = "n/a" if value != value else f"{value:.3f}"
            ax.text(col_idx, row_idx, text, ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def _write_search_candidates_csv(path, rows):
    fieldnames = [
        "stage",
        "stage_status",
        "final_label",
        "method_name",
        "selection_top_k",
        "segment_end",
        "fill_mode",
        "fill_rho",
        "n_steps",
        "segment_step_count",
        "accuracy_auc_mean",
        "accuracy_auc_std",
        "macro_f1_auc_mean",
        "best_accuracy_mean",
        "best_k_global",
        "runtime_mean",
        "runtime_std",
        "abs_error_mean",
        "pareto",
        "selected",
        "selection_reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_search_report_markdown(search_result, figures, search_candidates_csv):
    stage_a = search_result["stages"]["A"]
    stage_b = search_result["stages"]["B"]
    stage_c = search_result["stages"]["C"]
    lines = [
        "# ImageNet Feature-Selection Hyperparameter Search",
        "",
        "Staged Cheap-IG search on the ImageNet Sec. 5.2-style feature-selection benchmark.",
        "",
        "## Configuration",
        "",
        f"- layer_name=`{search_result['layer_name']}`",
        f"- search_tasks=`{[task['name'] for task in search_result['split']['search_tasks']]}`",
        f"- holdout_tasks=`{[task['name'] for task in search_result['split']['holdout_tasks']]}`",
        f"- k_values=`{search_result['search_space']['k_values']}`",
        f"- search_candidates_csv=`{search_candidates_csv}`",
        "",
        "## Stage A",
        "",
        _build_stage_table(stage_a["candidate_records"]),
        "",
        "## Stage B",
        "",
        _build_stage_table(stage_b["candidate_records"]),
        "",
        "## Stage C Finalists",
        "",
        _build_finalist_table(stage_c["finalists"]),
        "",
    ]
    for key in ("stage_a_heatmap", "stage_b_rho_heatmap", "nsteps_tradeoff", "search_pareto_scatter", "search_parameter_effects"):
        path = figures.get(key)
        if path:
            lines.extend([f"![]({seg._relative_markdown_path(path)})", ""])
    return "\n".join(lines)


def _build_holdout_report_markdown(search_result, figures):
    holdout = search_result["stages"]["holdout"]
    lines = [
        "# ImageNet Feature-Selection Holdout Evaluation",
        "",
        f"- holdout_tasks=`{[task['name'] for task in search_result['split']['holdout_tasks']]}`",
        f"- k_values=`{holdout['k_values']}`",
        "",
        "## Finalists",
        "",
        _build_finalist_table(search_result["finalists"]),
        "",
        "## Holdout Summary",
        "",
        _build_holdout_method_table(holdout["method_records"]),
        "",
        "## Pairwise Win Rate",
        "",
        _build_pairwise_table(holdout["pairwise_win_rate"]),
        "",
    ]
    for key in ("holdout_curves", "holdout_summary"):
        path = figures.get(key)
        if path:
            lines.extend([f"![]({seg._relative_markdown_path(path)})", ""])
    return "\n".join(lines)


def _build_stage_table(records):
    lines = [
        "| Method | Score | Std | Macro-F1 | Best acc | Best k | Runtime (s) | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in _sorted_candidate_records(records):
        lines.append(
            f"| {record['method_name']} | {seg._format_number(record['accuracy_auc_mean'])} | {seg._format_number(record['accuracy_auc_std'])} | {seg._format_number(record['macro_f1_auc_mean'])} | {seg._format_number(record['best_accuracy_mean'])} | {record['best_k_global']} | {seg._format_number(record['runtime_mean'])} | {record['stage_status']} |"
        )
    return "\n".join(lines)


def _build_finalist_table(finalists):
    lines = [
        "| Label | Method | Score | Best acc | Runtime (s) |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for finalist in finalists:
        lines.append(
            f"| {finalist['label']} | {finalist['method_name']} | {seg._format_number(finalist['accuracy_auc_mean'])} | {seg._format_number(finalist['best_accuracy_mean'])} | {seg._format_number(finalist['runtime_mean'])} |"
        )
    return "\n".join(lines)


def _build_holdout_method_table(records):
    lines = [
        "| Method | Label | Mean accuracy over k | Std | Best acc | Best k | Macro-F1 AUC | Runtime (s) | Delta vs IG | Delta vs NAA |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in sorted(records, key=lambda row: (-row["accuracy_auc"]["mean"], -row["best_accuracy"]["mean"], row["method_name"])):
        lines.append(
            f"| {record['method_name']} | {record.get('label') or ''} | {seg._format_number(record['accuracy_auc']['mean'])} | {seg._format_number(record['accuracy_auc']['std'])} | {seg._format_number(record['best_accuracy']['mean'])} | {record.get('best_k_global')} | {seg._format_number(record['macro_f1_auc']['mean'])} | {seg._format_number(record['runtime']['mean'])} | {seg._format_number(record.get('delta_vs_ig'))} | {seg._format_number(record.get('delta_vs_naa'))} |"
        )
    return "\n".join(lines)


def _build_pairwise_table(pairwise):
    method_names = list(pairwise)
    lines = [
        "| Left \\\\ Right | " + " | ".join(method_names) + " |",
        "| --- | " + " | ".join(["---:"] * len(method_names)) + " |",
    ]
    for left in method_names:
        row = [left]
        for right in method_names:
            value = pairwise[left][right]
            row.append("n/a" if value != value else f"{value:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _stage_json_payload(stage):
    return {
        "stage": stage["stage"],
        "title": stage["title"],
        "split": stage["split"],
        "task_names": stage["task_names"],
        "n_tasks": stage["n_tasks"],
        "k_values": stage["k_values"],
        "candidate_records": stage["candidate_records"],
        "finalists": stage.get("finalists"),
    }


def _stage_title(stage):
    return {
        "A": "topology screen",
        "B": "rho sweep",
        "C": "n_steps refinement",
    }.get(stage, str(stage))


def _config_key(config):
    return json.dumps(
        {
            "selection_mode": config["selection_mode"],
            "selection_top_k": int(config["selection_top_k"]),
            "segment_start": float(config["segment_start"]),
            "segment_end": float(config["segment_end"]),
            "fill_mode": config["fill_mode"],
            "fill_rho": None if config["fill_rho"] is None else float(config["fill_rho"]),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _config_step_key(config, n_steps):
    return json.dumps(
        {
            "config": json.loads(_config_key(config)),
            "n_steps": int(n_steps),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _pareto_keys(records):
    frontier = []
    for record in records:
        if not _is_finite(record["accuracy_auc_mean"]) or not _is_finite(record["runtime_mean"]):
            continue
        dominated = False
        for other in records:
            if other is record:
                continue
            if not _is_finite(other["accuracy_auc_mean"]) or not _is_finite(other["runtime_mean"]):
                continue
            better_or_equal = other["accuracy_auc_mean"] >= record["accuracy_auc_mean"] - DEFAULT_EPS and other["runtime_mean"] <= record["runtime_mean"] + DEFAULT_EPS
            strictly_better = other["accuracy_auc_mean"] > record["accuracy_auc_mean"] + DEFAULT_EPS or other["runtime_mean"] < record["runtime_mean"] - DEFAULT_EPS
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(record["config_step_key"])
    return frontier


def _sorted_candidate_records(records):
    return sorted(records, key=_candidate_sort_key)


def _candidate_sort_key(record):
    return (
        -float(record["accuracy_auc_mean"]) if _is_finite(record["accuracy_auc_mean"]) else float("inf"),
        float(record["runtime_mean"]) if _is_finite(record["runtime_mean"]) else float("inf"),
        -float(record["best_accuracy_mean"]) if _is_finite(record["best_accuracy_mean"]) else float("inf"),
        record["method_name"],
    )


def _candidate_rank_tuple(record):
    return (
        float(record["accuracy_auc_mean"]),
        -float(record["runtime_mean"]),
        float(record["best_accuracy_mean"]),
        -int(record["selection_top_k"]),
    )


def _segment_sample_count(segment_start, segment_end, n_steps):
    alphas = np.linspace(0.0, 1.0, int(n_steps), endpoint=True)
    return int(np.sum((alphas >= float(segment_start) - DEFAULT_EPS) & (alphas <= float(segment_end) + DEFAULT_EPS)))


def _short_config_label(record, *, include_n_steps):
    parts = [
        f"k={int(record['selection_top_k'])}",
        f"seg={float(record['segment_end']):g}",
        "zero" if record["fill_mode"] == "zero" else f"rho={float(record['fill_rho']):g}",
    ]
    if include_n_steps:
        parts.append(f"n={int(record['n_steps'])}")
    return ", ".join(parts)


def _display_name_with_label(method_name, label):
    return method_name if not label else f"{method_name} [{label}]"


def _placeholder_figure(title, text):
    fig, ax = plt.subplots(figsize=(6.5, 3.2), constrained_layout=True)
    ax.text(0.5, 0.5, text, ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    return fig


def _is_finite(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _pretty_json(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
