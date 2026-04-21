from __future__ import annotations

"""Simple AOPC benchmark for classifier attribution methods."""

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import IG, NAA, cheap_ig
from modules.method_timing_cache import current_device_label, image_signature, load_or_compute_cached_value


DEFAULT_CACHE_ROOT = "output/aopc_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_FILENAME = "aopc_report.md"
DEFAULT_SUMMARY_JSON = "aopc_summary.json"
DEFAULT_EPS = 1e-12
DEFAULT_UNIT_MODE = "neuron"
DEFAULT_PERTURBATION_MODE = "both"
DEFAULT_BUDGET_MODE = "percent_steps"
DEFAULT_BUDGET_STEP_FRACTION = 0.01
DEFAULT_BUDGET_NUM_STEPS = 100
DEFAULT_PERTURBATION_FRACTIONS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)


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
    for variant in cheap_ig_variants or []:
        specs.append(classifier_method_spec("cheap_ig", **variant))
    return specs


def benchmark_classifier_aopc(
    *,
    image_paths,
    method_specs,
    layer_name,
    n_steps,
    unit_mode=DEFAULT_UNIT_MODE,
    perturbation_mode=DEFAULT_PERTURBATION_MODE,
    budget_mode=DEFAULT_BUDGET_MODE,
    budget_step_fraction=DEFAULT_BUDGET_STEP_FRACTION,
    budget_num_steps=DEFAULT_BUDGET_NUM_STEPS,
    perturbation_counts=None,
    perturbation_fractions=None,
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
    image_paths = [str(Path(path)) for path in image_paths]
    unit_mode = _normalize_unit_mode(unit_mode)
    perturbation_mode = _normalize_perturbation_mode(perturbation_mode)
    budget_mode = _normalize_budget_mode(budget_mode)
    budget_step_fraction = _normalize_budget_step_fraction(budget_step_fraction)
    budget_num_steps = _normalize_budget_num_steps(budget_num_steps)
    perturbation_counts = _normalize_perturbation_counts(perturbation_counts)
    perturbation_fractions = _normalize_perturbation_fractions(perturbation_fractions)
    normalized_method_specs = _normalize_method_specs(method_specs)

    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = _prepare_existing_output_dir(target_dir)
        else:
            run_name = (
                f"aopc_classifier_{_safe_slug(unit_mode)}_{_safe_slug(layer_name)}"
                f"_mode_{_safe_slug(perturbation_mode)}_budget_{_safe_slug(budget_mode)}"
                f"_steps_{int(n_steps)}_images_{int(len(image_paths))}"
            )
            run_dir = _prepare_output_dir(output_dir, run_name)

    core_rows = []
    rows = []
    per_image = []

    for image_path in image_paths:
        core_record = _load_core_record(
            image_path=image_path,
            layer_name=layer_name,
            unit_mode=unit_mode,
            budget_mode=budget_mode,
            budget_step_fraction=budget_step_fraction,
            budget_num_steps=budget_num_steps,
            perturbation_counts=perturbation_counts,
            perturbation_fractions=perturbation_fractions,
            cache_root=cache_root,
            refresh=refresh_core,
        )
        if core_record.get("error") is not None or core_record.get("value") is None:
            raise RuntimeError(
                f"Failed to build AOPC core for {image_path}: {core_record.get('error')} "
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
                unit_mode=unit_mode,
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
                layer_name=layer_name,
                unit_mode=unit_mode,
                method_spec=method_spec,
                core_value=core_value,
                method_value=method_value,
                perturbation_mode=perturbation_mode,
                cache_root=cache_root,
                refresh=refresh_evaluations,
            )
            if evaluation_record.get("error") is not None or evaluation_record.get("value") is None:
                raise RuntimeError(
                    f"Failed to evaluate method {method_spec['name']} for {image_path}: {evaluation_record.get('error')} "
                    f"(cache={evaluation_record.get('cache_path')})"
                )
            evaluation = evaluation_record["value"]
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
        unit_mode=unit_mode,
        perturbation_mode=perturbation_mode,
        budget_mode=budget_mode,
        budget_step_fraction=budget_step_fraction,
        budget_num_steps=budget_num_steps,
        cache_root=cache_root,
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
        "task": "classifier",
        "image_paths": image_paths,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "unit_mode": unit_mode,
        "perturbation_mode": perturbation_mode,
        "budget_mode": budget_mode,
        "budget_step_fraction": float(budget_step_fraction),
        "budget_num_steps": int(budget_num_steps),
        "perturbation_counts": list(summary["perturbation_counts"]),
        "perturbation_fractions": list(summary["perturbation_fractions"]),
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
    image_path,
    layer_name,
    unit_mode,
    budget_mode,
    budget_step_fraction,
    budget_num_steps,
    perturbation_counts,
    perturbation_fractions,
    cache_root,
    refresh,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "layer_name": layer_name,
        "unit_mode": unit_mode,
        "budget_mode": budget_mode,
        "budget_step_fraction": float(budget_step_fraction),
        "budget_num_steps": int(budget_num_steps),
        "perturbation_counts": perturbation_counts,
        "perturbation_fractions": perturbation_fractions,
        "schema": 1,
    }
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="aopc_classifier_core",
        config=config,
        label=Path(image_path).stem,
        compute_fn=lambda: _compute_classifier_core_value(
            image_path=image_path,
            layer_name=layer_name,
            unit_mode=unit_mode,
            budget_mode=budget_mode,
            budget_step_fraction=budget_step_fraction,
            budget_num_steps=budget_num_steps,
            perturbation_counts=perturbation_counts,
            perturbation_fractions=perturbation_fractions,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _load_method_record(
    *,
    image_path,
    layer_name,
    n_steps,
    unit_mode,
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
        "unit_mode": unit_mode,
        "method_spec": method_spec,
        "top_n": int(top_n),
        "fd_eps": float(fd_eps),
        "clear_every": int(clear_every),
        "schema": 1,
    }
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="aopc_classifier_method",
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        compute_fn=lambda: _compute_classifier_method_value(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            unit_mode=unit_mode,
            method_spec=method_spec,
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
    layer_name,
    unit_mode,
    method_spec,
    core_value,
    method_value,
    perturbation_mode,
    cache_root,
    refresh,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "layer_name": layer_name,
        "unit_mode": unit_mode,
        "perturbation_mode": perturbation_mode,
        "method_spec": method_spec,
        "perturbation_counts": core_value["perturbation_counts"],
        "target_class": core_value["target_class"],
        "schema": 1,
    }
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="aopc_classifier_evaluation",
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}_eval",
        compute_fn=lambda: _compute_classifier_evaluation_value(
            image_path=image_path,
            layer_name=layer_name,
            unit_mode=unit_mode,
            core_value=core_value,
            method_value=method_value,
            perturbation_mode=perturbation_mode,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _compute_classifier_core_value(
    *,
    image_path,
    layer_name,
    unit_mode,
    budget_mode,
    budget_step_fraction,
    budget_num_steps,
    perturbation_counts,
    perturbation_fractions,
):
    _clear_all_backend_caches()
    runner = _ClassifierPerturbationRunner(image_path=image_path, layer_name=layer_name, unit_mode=unit_mode)
    try:
        counts = _resolve_perturbation_counts(
            n_units=runner.n_units,
            budget_mode=budget_mode,
            budget_step_fraction=budget_step_fraction,
            budget_num_steps=budget_num_steps,
            perturbation_counts=perturbation_counts,
            perturbation_fractions=perturbation_fractions,
        )
        fractions = [float(count) / max(1, int(runner.n_units)) for count in counts]
        reference_scores = _unit_reference_scores_from_tensor(runner.clean_act - runner.base_act, unit_mode=unit_mode)
        return {
            "task": "classifier",
            "image_path": image_path,
            "layer_name": layer_name,
            "unit_mode": unit_mode,
            "target_class": int(runner.target_class),
            "target_name": runner.target_name,
            "clean_score": float(runner.clean_score),
            "baseline_score": float(runner.baseline_score),
            "clean_delta": float(runner.clean_score - runner.baseline_score),
            "n_units_total": int(runner.n_units),
            "budget_mode": budget_mode,
            "budget_step_fraction": float(budget_step_fraction),
            "budget_num_steps": int(budget_num_steps),
            "n_budget_steps": int(len(counts)),
            "perturbation_counts": [int(v) for v in counts],
            "perturbation_fractions": [float(v) for v in fractions],
            "reference_scores": [float(v) for v in reference_scores.tolist()],
        }
    finally:
        runner.close()
        _clear_all_backend_caches()


def _compute_classifier_method_value(*, image_path, layer_name, n_steps, unit_mode, method_spec, top_n, fd_eps, clear_every):
    _clear_all_backend_caches()
    result = _run_classifier_method(
        image_path=image_path,
        layer_name=layer_name,
        n_steps=n_steps,
        unit_mode=unit_mode,
        method_spec=method_spec,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
    )
    _clear_all_backend_caches()
    return result


def _compute_classifier_evaluation_value(*, image_path, layer_name, unit_mode, core_value, method_value, perturbation_mode):
    _clear_all_backend_caches()
    runner = _ClassifierPerturbationRunner(
        image_path=image_path,
        layer_name=layer_name,
        unit_mode=unit_mode,
        target_class=core_value["target_class"],
    )
    try:
        return _evaluate_method_with_runner(core_value, method_value, runner, perturbation_mode=perturbation_mode)
    finally:
        runner.close()
        _clear_all_backend_caches()


def _run_classifier_method(*, image_path, layer_name, n_steps, unit_mode, method_spec, top_n, fd_eps, clear_every):
    kind = method_spec["kind"]
    common = {
        "image_path": image_path,
        "layer_name": layer_name,
        "n_steps": n_steps,
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
        "fill_mode",
        "fill_rho",
        "fill_beta",
        "filled_neurons",
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

        self.unit_labels = _build_single_tensor_unit_labels(self.layer_name, self.clean_act.shape, unit_mode=self.unit_mode)
        self.n_units = int(len(self.unit_labels))
        self._delta_cache = {
            ("deletion", ()): 0.0,
            ("insertion", ()): 0.0,
        }

    def _forward_with_capture(self, x_in):
        self.capture_hook.clear()
        out = IG.model(x_in)
        act = IG.unwrap_tensor(self.capture_hook.get())
        return out, act

    def score_delta_for_unit_indices(self, unit_indices, *, mode):
        mode = _normalize_single_perturbation_mode(mode)
        unit_indices = tuple(sorted(int(v) for v in unit_indices))
        cache_key = (mode, unit_indices)
        if cache_key in self._delta_cache:
            return self._delta_cache[cache_key]

        donor = self.base_act if mode == "deletion" else self.clean_act
        source_x = self.x if mode == "deletion" else self.x0
        index_tensor = torch.tensor(unit_indices, device=donor.device, dtype=torch.long)
        modules = dict(IG.model.named_modules())
        handle = modules[self.layer_name].register_forward_hook(
            lambda module, inp, out: _patch_single_tensor_units(
                out,
                donor,
                index_tensor,
                unit_mode=self.unit_mode,
            )
        )
        try:
            with torch.no_grad():
                out = IG.model(source_x)
                _, logits = IG.split_classifier_output(out)
                patched_score = float(logits[0, self.target_class].item())
            if mode == "deletion":
                delta = float(self.clean_score - patched_score)
            else:
                delta = float(patched_score - self.baseline_score)
        finally:
            handle.remove()
        self._delta_cache[cache_key] = delta
        return delta

    def score_drop_for_unit_indices(self, unit_indices):
        return self.score_delta_for_unit_indices(unit_indices, mode="deletion")

    def close(self):
        pass


def _evaluate_method_with_runner(core_value, method_value, runner, *, perturbation_mode):
    unit_scores = np.asarray(method_value["unit_scores"], dtype=np.float64)
    reference_scores = np.asarray(core_value["reference_scores"], dtype=np.float64)
    if unit_scores.size != reference_scores.size:
        raise ValueError(
            f"Method unit_scores has size={unit_scores.size}, but benchmark expects {reference_scores.size} units."
        )

    order = _stable_descending_order(unit_scores, reference_scores)
    counts = np.asarray(core_value["perturbation_counts"], dtype=np.int64)
    active_modes = _resolve_active_modes(perturbation_mode)
    clean_delta = float(core_value["clean_delta"])

    result = {}
    raw_scores = []
    normalized_scores = []
    for mode in active_modes:
        curve = []
        for count in counts.tolist():
            curve.append(float(runner.score_delta_for_unit_indices(order[: int(count)], mode=mode)))
        curve = np.asarray(curve, dtype=np.float64)
        if abs(clean_delta) <= DEFAULT_EPS:
            normalized_curve = np.full(curve.shape, np.nan, dtype=np.float64)
            aopc_normalized = float("nan")
        else:
            normalized_curve = curve / clean_delta
            aopc_normalized = float(np.nanmean(normalized_curve))
        aopc = float(curve.mean())
        result[f"{mode}_aopc"] = aopc
        result[f"{mode}_aopc_normalized"] = aopc_normalized
        result[f"{mode}_curve"] = [float(v) for v in curve.tolist()]
        result[f"{mode}_normalized_curve"] = [float(v) for v in normalized_curve.tolist()]
        raw_scores.append(aopc)
        normalized_scores.append(aopc_normalized)

    aopc = _mean_finite(raw_scores)
    aopc_normalized = _mean_finite(normalized_scores)
    head_n = min(20, int(order.size))
    return {
        "aopc": aopc,
        "aopc_normalized": aopc_normalized,
        "score": aopc_normalized,
        "perturbation_mode": perturbation_mode,
        "ordered_unit_indices_head": [int(v) for v in order[:head_n].tolist()],
        "ordered_unit_labels_head": [runner.unit_labels[int(v)] for v in order[:head_n].tolist()],
        "abs_error": float(method_value.get("abs_error", float("nan"))),
        "fx": float(method_value.get("fx", float("nan"))),
        "fx0": float(method_value.get("fx0", float("nan"))),
        "selected_neurons": method_value.get("selected_neurons"),
        **result,
    }


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
    unit_mode,
    perturbation_mode,
    budget_mode,
    budget_step_fraction,
    budget_num_steps,
    cache_root,
):
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method_name"], []).append(row)

    active_modes = _resolve_active_modes(perturbation_mode)
    method_summaries = {}
    for method_spec in method_specs:
        method_name = method_spec["name"]
        method_rows = by_method.get(method_name, [])
        summary = {
            "id": method_spec["id"],
            "kind": method_spec["kind"],
            "score": _stats_record([row["score"] for row in method_rows]),
            "aopc": _stats_record([row["aopc"] for row in method_rows]),
            "aopc_normalized": _stats_record([row["aopc_normalized"] for row in method_rows]),
            "abs_error": _stats_record([row["abs_error"] for row in method_rows]),
            "runtime_s": _stats_record([row["method_duration_s"] for row in method_rows]),
            "evaluation_runtime_s": _stats_record([row["evaluation_duration_s"] for row in method_rows]),
            "benchmark_runtime_s": _stats_record([row["benchmark_duration_s"] for row in method_rows]),
            "rank": _stats_record([row["rank"] for row in method_rows if row.get("rank") is not None]),
        }
        for mode in active_modes:
            summary[f"{mode}_aopc"] = _stats_record([row.get(f"{mode}_aopc") for row in method_rows])
            summary[f"{mode}_aopc_normalized"] = _stats_record([row.get(f"{mode}_aopc_normalized") for row in method_rows])
        method_summaries[method_name] = summary

    core_summary = {
        "clean_delta": _stats_record([row["core"]["clean_delta"] for row in core_rows]),
        "n_units_total": _stats_record([row["core"]["n_units_total"] for row in core_rows]),
        "n_budget_steps": _stats_record([row["core"].get("n_budget_steps") for row in core_rows]),
        "core_runtime_s": _stats_record([row["core_duration_s"] for row in core_rows]),
    }

    perturbation_counts = core_rows[0]["core"]["perturbation_counts"] if core_rows else []
    perturbation_fractions = core_rows[0]["core"]["perturbation_fractions"] if core_rows else []
    return {
        "task": "classifier",
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "unit_mode": unit_mode,
        "perturbation_mode": perturbation_mode,
        "budget_mode": budget_mode,
        "budget_step_fraction": float(budget_step_fraction),
        "budget_num_steps": int(budget_num_steps),
        "n_images": int(len(core_rows)),
        "cache_root": str(cache_root),
        "n_budget_steps": int(len(perturbation_counts)),
        "perturbation_counts": [int(v) for v in perturbation_counts],
        "perturbation_fractions": [float(v) for v in perturbation_fractions],
        "method_summaries": method_summaries,
        "core_summary": core_summary,
        "pairwise_win_rate": _pairwise_win_rate(rows, method_specs),
    }


def _render_and_save_report_figures(*, run_dir, summary, rows, core_rows, method_specs):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {}

    fig = _plot_aopc_summary(summary, method_specs)
    path = figure_dir / "aopc_summary.png"
    _save_figure(fig, path)
    figures["aopc_summary"] = str(path)

    fig = _plot_aopc_distributions(rows, method_specs)
    path = figure_dir / "aopc_distributions.png"
    _save_figure(fig, path)
    figures["aopc_distributions"] = str(path)

    fig = _plot_mean_curves(rows, core_rows, method_specs)
    path = figure_dir / "aopc_curves.png"
    _save_figure(fig, path)
    figures["aopc_curves"] = str(path)

    fig = _plot_pairwise_win_heatmap(summary, method_specs)
    path = figure_dir / "aopc_pairwise_wins.png"
    _save_figure(fig, path)
    figures["aopc_pairwise_wins"] = str(path)

    return figures


def _build_report_markdown(summary, rows, core_rows, *, method_specs, figures):
    lines = [
        "# AOPC Benchmark",
        "",
        f"- task=`{summary['task']}`",
        f"- layer_name=`{summary['layer_name']}`",
        f"- n_steps={summary['n_steps']}",
        f"- unit_mode=`{summary['unit_mode']}`",
        f"- perturbation_mode=`{summary['perturbation_mode']}`",
        f"- budget_mode=`{summary['budget_mode']}`",
        f"- budget_step_fraction={summary['budget_step_fraction']:.4f}",
        f"- budget_num_steps={summary['budget_num_steps']}",
        f"- effective_budget_steps={summary['n_budget_steps']}",
        f"- n_images={summary['n_images']}",
        f"- cache_root=`{summary['cache_root']}`",
    ]

    active_modes = _resolve_active_modes(summary["perturbation_mode"])
    if len(active_modes) == 1:
        metric_headers = [_score_label(summary["perturbation_mode"])]
    else:
        metric_headers = ["Score"]
        for mode in active_modes:
            metric_headers.append(f"{_mode_label(mode)} / clean_delta")

    lines.extend(
        [
            "",
            "## Aggregate Summary",
            "",
            "| Method | " + " | ".join(metric_headers) + " | Mean Rank | Attr Runtime (s) | Eval Runtime (s) | Abs Error |",
            "| --- | " + " | ".join("---:" for _ in metric_headers) + " | ---: | ---: | ---: | ---: |",
        ]
    )

    for method_spec in method_specs:
        stats = summary["method_summaries"][method_spec["name"]]
        if len(active_modes) == 1:
            metric_values = [_format_stats(stats["score"])]
        else:
            metric_values = [_format_stats(stats["score"])]
            for mode in active_modes:
                metric_values.append(_format_stats(stats[f"{mode}_aopc_normalized"]))
        lines.append(
            f"| {method_spec['name']} | "
            + " | ".join(metric_values)
            + " | "
            + " | ".join(
                [
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
            f"| clean_delta | {_format_stats(core_summary['clean_delta'])} |",
            f"| n_units_total | {_format_stats(core_summary['n_units_total'])} |",
            f"| n_budget_steps | {_format_stats(core_summary['n_budget_steps'])} |",
            f"| core_runtime_s | {_format_stats(core_summary['core_runtime_s'])} |",
        ]
    )

    if figures:
        lines.extend(["", "## Figures", ""])
        for key in ("aopc_summary", "aopc_distributions", "aopc_curves", "aopc_pairwise_wins"):
            path = figures.get(key)
            if path:
                lines.extend([f"### {key}", "", f"![]({_relative_markdown_path(path)})", ""])

    lines.extend(
        [
            "## Per-Image Score",
            "",
            _build_per_image_markdown_table(rows, method_specs),
            "",
            "## Pairwise Win Rate",
            "",
            _build_pairwise_markdown_table(summary["pairwise_win_rate"], method_specs),
        ]
    )
    return "\n".join(lines)


def _plot_aopc_summary(summary, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    means = [summary["method_summaries"][name]["score"]["mean"] for name in method_names]
    stds = [summary["method_summaries"][name]["score"]["std"] for name in method_names]
    ranks = [summary["method_summaries"][name]["rank"]["mean"] for name in method_names]
    x = np.arange(len(method_names))

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(x, means, yerr=stds, capsize=6, color="#4c78a8")
    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylabel(_score_label(summary["perturbation_mode"]))
    ax.set_title(
        f"Mean {_score_label(summary['perturbation_mode'])} by method, "
        f"n={summary['n_images']}, unit={summary['unit_mode']}"
    )
    ax.grid(axis="y", alpha=0.3)
    for idx, (mean_value, rank_value) in enumerate(zip(means, ranks)):
        if mean_value == mean_value:
            label = f"rank={rank_value:.2f}" if rank_value == rank_value else "rank=n/a"
            ax.text(idx, mean_value + max(0.01, stds[idx] * 0.3 + 0.01), label, ha="center", va="bottom", fontsize=9)
    return fig


def _plot_aopc_distributions(rows, method_specs):
    method_names = [spec["name"] for spec in method_specs]
    values = []
    for name in method_names:
        current = [row["score"] for row in rows if row["method_name"] == name and row["score"] == row["score"]]
        values.append(current if current else [float("nan")])

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    box = ax.boxplot(values, patch_artist=True, labels=method_names)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(method_names))))
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    mode = rows[0]["perturbation_mode"] if rows else DEFAULT_PERTURBATION_MODE
    ax.set_ylabel(_score_label(mode))
    ax.set_title(f"{_score_label(mode)} distribution by method")
    ax.grid(axis="y", alpha=0.3)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")
    return fig


def _plot_mean_curves(rows, core_rows, method_specs):
    if not core_rows:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        ax.set_axis_off()
        return fig

    perturbation_mode = rows[0]["perturbation_mode"] if rows else DEFAULT_PERTURBATION_MODE
    active_modes = _resolve_active_modes(perturbation_mode)
    counts = np.asarray(core_rows[0]["core"]["perturbation_counts"], dtype=np.int64)
    x = np.arange(len(counts))
    tick_positions, tick_labels = _curve_tick_labels(
        counts,
        np.asarray(core_rows[0]["core"]["perturbation_fractions"], dtype=np.float64),
    )
    fig, axes = plt.subplots(2, len(active_modes), figsize=(7 * len(active_modes), 8), constrained_layout=True, squeeze=False)

    for col_idx, mode in enumerate(active_modes):
        for color_idx, spec in enumerate(method_specs):
            method_rows = [row for row in rows if row["method_name"] == spec["name"]]
            curves = np.asarray([row[f"{mode}_curve"] for row in method_rows], dtype=np.float64)
            normalized_curves = np.asarray([row[f"{mode}_normalized_curve"] for row in method_rows], dtype=np.float64)
            if curves.size == 0:
                continue
            color = plt.cm.tab10(color_idx % 10)
            axes[0, col_idx].plot(x, np.nanmean(curves, axis=0), label=spec["name"], color=color, marker="o")
            axes[1, col_idx].plot(x, np.nanmean(normalized_curves, axis=0), label=spec["name"], color=color, marker="o")

        axes[0, col_idx].set_title(f"{_mode_label(mode)} curve, unit={core_rows[0]['core']['unit_mode']}")
        axes[0, col_idx].set_xticks(tick_positions)
        axes[0, col_idx].set_xticklabels(tick_labels, rotation=25, ha="right")
        axes[0, col_idx].set_xlabel("Perturbed units")
        axes[0, col_idx].set_ylabel(_raw_curve_label(mode))
        axes[0, col_idx].grid(True, alpha=0.3)
        axes[0, col_idx].legend()

        axes[1, col_idx].set_title(f"{_mode_label(mode)} curve / clean delta, unit={core_rows[0]['core']['unit_mode']}")
        axes[1, col_idx].set_xticks(tick_positions)
        axes[1, col_idx].set_xticklabels(tick_labels, rotation=25, ha="right")
        axes[1, col_idx].set_xlabel("Perturbed units")
        axes[1, col_idx].set_ylabel(_normalized_curve_label(mode))
        axes[1, col_idx].grid(True, alpha=0.3)
        axes[1, col_idx].legend()
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
        by_image.setdefault(row["image_name"], {})[row["method_name"]] = row["score"]
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


def _stable_descending_order(unit_scores, reference_scores):
    scores = np.asarray(unit_scores, dtype=np.float64).copy()
    refs = np.asarray(reference_scores, dtype=np.float64).copy()
    idx = np.arange(scores.size, dtype=np.int64)
    scores[~np.isfinite(scores)] = -np.inf
    refs[~np.isfinite(refs)] = -np.inf
    order = np.lexsort((idx, -refs, -scores))
    return order.astype(np.int64, copy=False)


def _resolve_perturbation_counts(
    *,
    n_units,
    budget_mode,
    budget_step_fraction,
    budget_num_steps,
    perturbation_counts,
    perturbation_fractions,
):
    counts = []
    for value in perturbation_counts or ():
        counts.append(_clip_count(int(value), n_units))
    for fraction in perturbation_fractions or ():
        counts.append(_clip_count(int(np.ceil(float(fraction) * float(n_units))), n_units))
    if not counts:
        if budget_mode == "percent_steps":
            for step_idx in range(1, int(budget_num_steps) + 1):
                counts.append(
                    _clip_count(
                        int(np.ceil(float(step_idx) * float(budget_step_fraction) * float(n_units))),
                        n_units,
                    )
                )
        else:
            for fraction in DEFAULT_PERTURBATION_FRACTIONS:
                counts.append(_clip_count(int(np.ceil(float(fraction) * float(n_units))), n_units))
    return _unique_preserve_order(counts)


def _clip_count(count, n_units):
    return max(1, min(int(count), int(n_units)))


def _unique_preserve_order(values):
    seen = set()
    result = []
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _curve_tick_labels(counts, fractions, max_ticks=12):
    counts = np.asarray(counts, dtype=np.int64)
    fractions = np.asarray(fractions, dtype=np.float64)
    if counts.size == 0:
        return np.asarray([], dtype=np.int64), []
    tick_positions = np.linspace(0, counts.size - 1, num=min(max_ticks, counts.size), dtype=int)
    tick_positions = np.unique(tick_positions)
    labels = []
    for pos in tick_positions.tolist():
        frac = float(fractions[pos]) if pos < fractions.size else float("nan")
        pct = int(round(frac * 100.0)) if frac == frac else None
        count = int(counts[pos])
        if pct is not None and abs(frac * 100.0 - pct) < 1e-6:
            labels.append(f"{pct}%\n({count})")
        else:
            labels.append(str(count))
    return tick_positions, labels


def _normalize_unit_mode(unit_mode):
    unit_mode = str(unit_mode).strip().lower()
    if unit_mode not in {"filter", "spatial_cell", "neuron"}:
        raise ValueError(f"Unsupported unit_mode: {unit_mode}. Expected 'filter', 'spatial_cell', or 'neuron'.")
    return unit_mode


def _normalize_budget_mode(budget_mode):
    budget_mode = str(budget_mode).strip().lower()
    if budget_mode not in {"percent_steps", "fractions"}:
        raise ValueError(f"Unsupported budget_mode: {budget_mode}. Expected 'percent_steps' or 'fractions'.")
    return budget_mode


def _normalize_budget_step_fraction(budget_step_fraction):
    value = float(budget_step_fraction)
    if not (0.0 < value <= 1.0):
        raise ValueError(f"budget_step_fraction must be in (0, 1], got {value}.")
    return value


def _normalize_budget_num_steps(budget_num_steps):
    value = int(budget_num_steps)
    if value < 1:
        raise ValueError(f"budget_num_steps must be >= 1, got {value}.")
    return value


def _normalize_perturbation_mode(perturbation_mode):
    perturbation_mode = str(perturbation_mode).strip().lower()
    if perturbation_mode not in {"deletion", "insertion", "both"}:
        raise ValueError(
            f"Unsupported perturbation_mode: {perturbation_mode}. Expected 'deletion', 'insertion', or 'both'."
        )
    return perturbation_mode


def _normalize_single_perturbation_mode(mode):
    mode = str(mode).strip().lower()
    if mode not in {"deletion", "insertion"}:
        raise ValueError(f"Unsupported perturbation mode: {mode}. Expected 'deletion' or 'insertion'.")
    return mode


def _resolve_active_modes(perturbation_mode):
    perturbation_mode = _normalize_perturbation_mode(perturbation_mode)
    if perturbation_mode == "both":
        return ("deletion", "insertion")
    return (perturbation_mode,)


def _mean_finite(values):
    numeric = np.asarray([float(v) for v in values if v is not None and float(v) == float(v)], dtype=np.float64)
    if numeric.size == 0:
        return float("nan")
    return float(numeric.mean())


def _mode_label(mode):
    mode = _normalize_single_perturbation_mode(mode)
    return "Deletion" if mode == "deletion" else "Insertion"


def _score_label(perturbation_mode):
    perturbation_mode = _normalize_perturbation_mode(perturbation_mode)
    if perturbation_mode == "deletion":
        return "Deletion AOPC / clean_delta"
    if perturbation_mode == "insertion":
        return "Insertion AOPC / clean_delta"
    return "Mean(Deletion, Insertion) AOPC / clean_delta"


def _raw_curve_label(mode):
    mode = _normalize_single_perturbation_mode(mode)
    return "Score drop" if mode == "deletion" else "Score rise"


def _normalized_curve_label(mode):
    mode = _normalize_single_perturbation_mode(mode)
    return "Normalized drop" if mode == "deletion" else "Normalized rise"


def _normalize_perturbation_counts(perturbation_counts):
    if perturbation_counts is None:
        return None
    values = sorted({int(value) for value in perturbation_counts if int(value) > 0})
    if not values:
        raise ValueError("perturbation_counts must contain at least one positive integer.")
    return tuple(values)


def _normalize_perturbation_fractions(perturbation_fractions):
    if perturbation_fractions is None:
        return None
    values = []
    for raw_value in perturbation_fractions:
        value = float(raw_value)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Perturbation fraction must be in (0, 1], got {value}.")
        values.append(value)
    normalized = tuple(sorted(set(values)))
    if not normalized:
        raise ValueError("perturbation_fractions must contain at least one value.")
    return normalized


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
    if unit_mode == "neuron":
        return arr.reshape(-1)
    if arr.ndim == 1:
        return arr
    return arr.reshape(arr.shape[0], -1).sum(axis=0)


def _unit_reference_scores_from_tensor(tensor, *, unit_mode):
    unit_mode = _normalize_unit_mode(unit_mode)
    arr = tensor[0].detach().cpu().numpy().astype(np.float64, copy=False)
    if unit_mode == "filter":
        if arr.ndim == 1:
            return np.abs(arr)
        return np.abs(arr).reshape(arr.shape[0], -1).sum(axis=1)
    if unit_mode == "neuron":
        return np.abs(arr).reshape(-1)
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


def _build_single_tensor_unit_labels(layer_name, tensor_shape, *, unit_mode):
    fake = torch.zeros(tensor_shape)
    labels = []
    per_sample = fake[0]
    if unit_mode == "filter":
        for local_idx in range(int(per_sample.shape[0])):
            labels.append(f"{layer_name}:f{local_idx}")
        return labels
    if unit_mode == "neuron":
        if per_sample.ndim == 1:
            for local_idx in range(int(per_sample.shape[0])):
                labels.append(f"{layer_name}:n{local_idx}")
            return labels
        neuron_shape = tuple(int(dim) for dim in per_sample.shape)
        for flat_idx in range(int(np.prod(neuron_shape, dtype=np.int64))):
            coords = np.unravel_index(flat_idx, neuron_shape)
            labels.append(f"{layer_name}:n{coords}")
        return labels
    if per_sample.ndim == 1:
        for local_idx in range(int(per_sample.shape[0])):
            labels.append(f"{layer_name}:n{local_idx}")
        return labels
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
    if unit_mode == "neuron":
        flat_patched = patched.reshape(patched.shape[0], -1)
        flat_baseline = baseline.reshape(baseline.shape[0], -1)
        flat_patched[:, index_tensor] = flat_baseline[:, index_tensor]
        return flat_patched.reshape_as(patched)
    flat_patched = patched.reshape(patched.shape[0], patched.shape[1], -1)
    flat_baseline = baseline.reshape(baseline.shape[0], baseline.shape[1], -1)
    flat_patched[:, :, index_tensor] = flat_baseline[:, :, index_tensor]
    return flat_patched.reshape_as(patched)


def _normalize_method_specs(method_specs):
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
    fill_mode = spec.get("fill_mode", "zero")
    if str(fill_mode) == "zero":
        fill_fragment = ""
    else:
        fill_fragment = f"/fill-{fill_mode}-rho{float(spec.get('fill_rho', 0.8)):g}"
    return f"cheap-ig[{seg_start:g},{seg_end:g}]/{selection_mode}/k{int(selection_top_k)}{fill_fragment}"


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
    import json

    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True)
