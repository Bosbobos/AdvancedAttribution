from __future__ import annotations

"""Monte-Carlo Shapley audit benchmark for neuron attribution methods."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import IG
from modules import aopc_benchmark as aopc
from modules import alpha_segment_benchmark as seg
from modules.baseline_utils import DEFAULT_BLUR_SIGMA
from modules.method_timing_cache import current_device_label, image_signature, load_or_compute_cached_value

try:
    from scipy.stats import rankdata, spearmanr

    SCIPY_STATS_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime environment dependent.
    rankdata = None
    spearmanr = None
    SCIPY_STATS_IMPORT_ERROR = exc


DEFAULT_CACHE_ROOT = "output/neuron_shapley_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_FILENAME = "neuron_shapley_report.md"
DEFAULT_SUMMARY_JSON = "neuron_shapley_summary.json"
DEFAULT_LAYER_NAME = "model.6"
DEFAULT_N_STEPS = 128
DEFAULT_UNIT_MODE = "spatial_cell"
DEFAULT_POOL_SIZE = 196
DEFAULT_NUM_PERMUTATIONS = 128
DEFAULT_POOL_SELECTION_MODE = "active_random"
DEFAULT_ACTIVE_MIN_ABS_DELTA = 1e-6
DEFAULT_STRATIFIED_NUM_BINS = 4
DEFAULT_ORACLE_IMPUTER_KIND = "black_act"
DEFAULT_RANDOM_SEED = 0
DEFAULT_PREVIEW_IMAGES = 5
DEFAULT_NDCG_K = 10
DEFAULT_RECALL_K = 10
DEFAULT_SIGN_THRESHOLD_RATIO = 0.05


def classifier_method_spec(kind, name=None, **kwargs):
    spec = {"kind": str(kind)}
    spec.update(kwargs)
    spec["name"] = str(name) if name is not None else _default_method_name(spec)
    return spec


def default_classifier_method_specs():
    return [
        classifier_method_spec("ig", name="IG", segment_start=0.0, segment_end=1.0),
        classifier_method_spec("naa", name="NAA", segment_start=0.0, segment_end=1.0),
        classifier_method_spec(
            "cheap_ig",
            name="Cheap-IG+[0,0.1]/k8000/zero",
            segment_start=0.0,
            segment_end=0.1,
            selection_mode="positive",
            selection_top_k=8000,
            fill_mode="zero",
            fill_rho=0.8,
        ),
    ]


def benchmark_classifier_neuron_shapley(
    *,
    image_paths,
    method_specs,
    layer_name=DEFAULT_LAYER_NAME,
    n_steps=DEFAULT_N_STEPS,
    unit_mode=DEFAULT_UNIT_MODE,
    pool_size=DEFAULT_POOL_SIZE,
    num_permutations=DEFAULT_NUM_PERMUTATIONS,
    pool_selection_mode=DEFAULT_POOL_SELECTION_MODE,
    active_min_abs_delta=DEFAULT_ACTIVE_MIN_ABS_DELTA,
    stratified_num_bins=DEFAULT_STRATIFIED_NUM_BINS,
    oracle_imputer_kind=DEFAULT_ORACLE_IMPUTER_KIND,
    random_seed=DEFAULT_RANDOM_SEED,
    preview_images=DEFAULT_PREVIEW_IMAGES,
    ndcg_k=DEFAULT_NDCG_K,
    recall_k=DEFAULT_RECALL_K,
    sign_threshold_ratio=DEFAULT_SIGN_THRESHOLD_RATIO,
    blur_sigma=DEFAULT_BLUR_SIGMA,
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
    refresh_oracle=False,
    refresh_methods=False,
    refresh_evaluations=False,
    verbose=False,
):
    _require_scipy_stats()

    image_paths = [str(Path(path)) for path in image_paths]
    normalized_method_specs = seg._normalize_method_specs(method_specs)
    preview_images = int(max(0, min(int(preview_images), len(image_paths))))
    unit_mode = aopc._normalize_unit_mode(unit_mode)
    if unit_mode not in {"neuron", "spatial_cell"}:
        raise ValueError("unit_mode must be `neuron` or `spatial_cell`.")
    pool_size = int(pool_size)
    num_permutations = int(num_permutations)
    stratified_num_bins = int(max(2, int(stratified_num_bins)))
    random_seed = int(random_seed)
    ndcg_k = int(ndcg_k)
    recall_k = int(recall_k)
    sign_threshold_ratio = float(sign_threshold_ratio)
    active_min_abs_delta = float(active_min_abs_delta)
    oracle_imputer_kind = str(oracle_imputer_kind)
    if oracle_imputer_kind not in seg.DEFAULT_DONOR_KINDS:
        raise ValueError(
            f"Unsupported oracle_imputer_kind: {oracle_imputer_kind}. "
            f"Supported values: {list(seg.DEFAULT_DONOR_KINDS)}"
        )
    if pool_selection_mode not in {"active_random", "stratified_activation_change"}:
        raise ValueError(
            "pool_selection_mode must be one of "
            "`active_random`, `stratified_activation_change`."
        )

    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = seg._prepare_existing_output_dir(target_dir)
        else:
            run_name = (
                f"neuron_shapley_{seg._safe_slug(layer_name)}"
                f"_{seg._safe_slug(unit_mode)}"
                f"_k{pool_size}_m{num_permutations}_images_{int(len(image_paths))}"
            )
            run_dir = seg._prepare_output_dir(output_dir, run_name)

    core_rows = []
    oracle_rows = []
    method_rows = []
    rows = []

    for image_index, image_path in enumerate(image_paths):
        core_record = seg._load_core_record(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            blur_sigma=blur_sigma,
            budget_percentiles=[1, 20],
            cache_root=cache_root,
            refresh=refresh_core,
        )
        if core_record.get("error") is not None or core_record.get("value") is None:
            raise RuntimeError(
                f"Failed to build neuron Shapley core for {image_path}: {core_record.get('error')} "
                f"(cache={core_record.get('cache_path')})"
            )
        core_value = core_record["value"]
        core_rows.append(
            {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "core_cache_path": core_record["cache_path"],
                "core_from_cache": core_record["from_cache"],
                "core_duration_s": core_record["duration_s"],
                "core": core_value,
            }
        )

        oracle_record = _load_oracle_record(
            image_path=image_path,
            image_index=image_index,
            layer_name=layer_name,
            n_steps=n_steps,
            unit_mode=unit_mode,
            pool_size=pool_size,
            num_permutations=num_permutations,
            pool_selection_mode=pool_selection_mode,
            active_min_abs_delta=active_min_abs_delta,
            stratified_num_bins=stratified_num_bins,
            oracle_imputer_kind=oracle_imputer_kind,
            random_seed=random_seed,
            core_value=core_value,
            cache_root=cache_root,
            refresh=refresh_oracle,
        )
        if oracle_record.get("error") is not None or oracle_record.get("value") is None:
            raise RuntimeError(
                f"Failed to build neuron Shapley oracle for {image_path}: {oracle_record.get('error')} "
                f"(cache={oracle_record.get('cache_path')})"
            )
        oracle_value = oracle_record["value"]
        oracle_rows.append(
            {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "oracle_cache_path": oracle_record["cache_path"],
                "oracle_from_cache": oracle_record["from_cache"],
                "oracle_duration_s": oracle_record["duration_s"],
                **oracle_value,
            }
        )

        for method_spec in normalized_method_specs:
            method_record = seg._load_method_record(
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
            method_rows.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
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
                layer_name=layer_name,
                n_steps=n_steps,
                unit_mode=unit_mode,
                method_spec=method_spec,
                method_value=method_value,
                oracle_value=oracle_value,
                ndcg_k=ndcg_k,
                recall_k=recall_k,
                sign_threshold_ratio=sign_threshold_ratio,
                cache_root=cache_root,
                refresh=refresh_evaluations,
            )
            if evaluation_record.get("error") is not None or evaluation_record.get("value") is None:
                raise RuntimeError(
                    f"Failed to evaluate method {method_spec['name']} for {image_path}: {evaluation_record.get('error')} "
                    f"(cache={evaluation_record.get('cache_path')})"
                )
            evaluation = evaluation_record["value"]
            rows.append(
                {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "method_name": method_spec["name"],
                    "method_id": method_spec["id"],
                    "kind": method_spec["kind"],
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
            )

    summary = _build_summary(
        rows=rows,
        core_rows=core_rows,
        oracle_rows=oracle_rows,
        method_rows=method_rows,
        method_specs=normalized_method_specs,
        layer_name=layer_name,
        n_steps=n_steps,
        unit_mode=unit_mode,
        pool_size=pool_size,
        num_permutations=num_permutations,
        pool_selection_mode=pool_selection_mode,
        active_min_abs_delta=active_min_abs_delta,
        stratified_num_bins=stratified_num_bins,
        oracle_imputer_kind=oracle_imputer_kind,
        random_seed=random_seed,
        ndcg_k=ndcg_k,
        recall_k=recall_k,
        sign_threshold_ratio=sign_threshold_ratio,
        cache_root=cache_root,
        preview_image_paths=image_paths[:preview_images],
    )

    figures = {}
    preview_sections = []
    if run_dir is not None:
        figures = _render_and_save_report_figures(
            run_dir=run_dir,
            summary=summary,
            rows=rows,
        )
        preview_sections = _render_preview_sections(
            run_dir=run_dir,
            summary=summary,
            oracle_rows=oracle_rows,
            method_rows=method_rows,
        )
        report_md = _build_report_markdown(summary, figures=figures, preview_sections=preview_sections)
        report_path = run_dir / report_filename
        report_path.write_text(report_md + "\n", encoding="utf-8")
        summary_path = run_dir / summary_filename
        summary_path.write_text(
            seg._pretty_json(
                {
                    "summary": summary,
                    "rows": rows,
                    "method_rows": method_rows,
                    "oracle_rows": oracle_rows,
                    "core_rows": core_rows,
                    "preview_sections": preview_sections,
                }
            ),
            encoding="utf-8",
        )
    else:
        report_md = _build_report_markdown(summary, figures={}, preview_sections=[])
        report_path = None
        summary_path = None

    return {
        "task": "classifier",
        "image_paths": image_paths,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "pool_size": int(pool_size),
        "num_permutations": int(num_permutations),
        "pool_selection_mode": pool_selection_mode,
        "active_min_abs_delta": float(active_min_abs_delta),
        "stratified_num_bins": int(stratified_num_bins),
        "oracle_imputer_kind": oracle_imputer_kind,
        "random_seed": int(random_seed),
        "ndcg_k": int(ndcg_k),
        "recall_k": int(recall_k),
        "sign_threshold_ratio": float(sign_threshold_ratio),
        "method_specs": normalized_method_specs,
        "rows": rows,
        "method_rows": method_rows,
        "oracle_rows": oracle_rows,
        "core_rows": core_rows,
        "summary": summary,
        "report_markdown": report_md,
        "report_path": str(report_path) if report_path is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "figures": figures,
        "preview_sections": preview_sections,
        "output_dir": str(run_dir) if run_dir is not None else None,
        "cache_root": str(cache_root),
    }


def render_neuron_shapley_report(result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = _render_and_save_report_figures(
        run_dir=output_dir,
        summary=result["summary"],
        rows=result["rows"],
    )
    preview_sections = _render_preview_sections(
        run_dir=output_dir,
        summary=result["summary"],
        oracle_rows=result["oracle_rows"],
        method_rows=result["method_rows"],
    )
    report_md = _build_report_markdown(result["summary"], figures=figures, preview_sections=preview_sections)
    report_path = output_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text(report_md + "\n", encoding="utf-8")
    summary_path = output_dir / DEFAULT_SUMMARY_JSON
    summary_path.write_text(
        seg._pretty_json(
            {
                "summary": result["summary"],
                "rows": result["rows"],
                "method_rows": result["method_rows"],
                "oracle_rows": result["oracle_rows"],
                "core_rows": result["core_rows"],
                "preview_sections": preview_sections,
            }
        ),
        encoding="utf-8",
    )
    return {
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "figures": figures,
        "preview_sections": preview_sections,
        "output_dir": str(output_dir),
    }


def _load_oracle_record(
    *,
    image_path,
    image_index,
    layer_name,
    n_steps,
    unit_mode,
    pool_size,
    num_permutations,
    pool_selection_mode,
    active_min_abs_delta,
    stratified_num_bins,
    oracle_imputer_kind,
    random_seed,
    core_value,
    cache_root,
    refresh,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "unit_mode": str(unit_mode),
        "pool_size": int(pool_size),
        "num_permutations": int(num_permutations),
        "pool_selection_mode": str(pool_selection_mode),
        "active_min_abs_delta": float(active_min_abs_delta),
        "stratified_num_bins": int(stratified_num_bins),
        "oracle_imputer_kind": str(oracle_imputer_kind),
        "random_seed": int(random_seed),
        "target_class": int(core_value["target_class"]),
        "schema": 1,
    }
    sidecar_key = seg._config_hash(config)
    image_seed = _image_seed(random_seed, image_index, image_path)
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="neuron_shapley_oracle",
        config=config,
        label=f"{Path(image_path).stem}_{pool_selection_mode}_k{pool_size}_m{num_permutations}",
        compute_fn=lambda: _compute_classifier_oracle_value(
            image_path=image_path,
            layer_name=layer_name,
            core_value=core_value,
            unit_mode=unit_mode,
            pool_size=pool_size,
            num_permutations=num_permutations,
            pool_selection_mode=pool_selection_mode,
            active_min_abs_delta=active_min_abs_delta,
            stratified_num_bins=stratified_num_bins,
            oracle_imputer_kind=oracle_imputer_kind,
            image_seed=image_seed,
            cache_root=cache_root,
            sidecar_key=sidecar_key,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _load_evaluation_record(
    *,
    image_path,
    layer_name,
    n_steps,
    unit_mode,
    method_spec,
    method_value,
    oracle_value,
    ndcg_k,
    recall_k,
    sign_threshold_ratio,
    cache_root,
    refresh,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "unit_mode": str(unit_mode),
        "method_spec": method_spec,
        "pool_indices_path": oracle_value["pool_indices_path"],
        "oracle_values_path": oracle_value["oracle_values_path"],
        "ndcg_k": int(ndcg_k),
        "recall_k": int(recall_k),
        "sign_threshold_ratio": float(sign_threshold_ratio),
        "schema": 1,
    }
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="neuron_shapley_evaluation",
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        compute_fn=lambda: _compute_classifier_evaluation_value(
            method_value=method_value,
            oracle_value=oracle_value,
            unit_mode=unit_mode,
            ndcg_k=ndcg_k,
            recall_k=recall_k,
            sign_threshold_ratio=sign_threshold_ratio,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _compute_classifier_oracle_value(
    *,
    image_path,
    layer_name,
    core_value,
    unit_mode,
    pool_size,
    num_permutations,
    pool_selection_mode,
    active_min_abs_delta,
    stratified_num_bins,
    oracle_imputer_kind,
    image_seed,
    cache_root,
    sidecar_key,
):
    runner = _ClassifierShapleyRunner(
        image_path=image_path,
        layer_name=layer_name,
        core_value=core_value,
        unit_mode=unit_mode,
    )
    try:
        delta = _reference_unit_scores(runner.clean_act_tensor - runner.black_act_tensor, unit_mode=unit_mode)
        rng = np.random.default_rng(int(image_seed))
        pool_indices = _select_candidate_pool(
            delta=delta,
            pool_size=pool_size,
            mode=pool_selection_mode,
            rng=rng,
            active_min_abs_delta=active_min_abs_delta,
            stratified_num_bins=stratified_num_bins,
        )
        shapley_values = _compute_mc_shapley(
            runner=runner,
            pool_indices=pool_indices,
            donor_kind=oracle_imputer_kind,
            num_permutations=num_permutations,
            rng=np.random.default_rng(int(image_seed) + 1),
        )
        oracle_overlay = _build_oracle_overlay(
            shapley_values=shapley_values,
            pool_indices=pool_indices,
            activation_shape=tuple(int(v) for v in core_value["activation_shape"]),
            image_shape=(int(core_value["image_height"]), int(core_value["image_width"])),
            unit_mode=unit_mode,
        )
        pool_indices_path = seg._array_sidecar_path(
            cache_root=cache_root,
            namespace="neuron_shapley_oracle",
            label=Path(image_path).stem,
            sidecar_key=sidecar_key,
            kind="pool_indices",
        )
        oracle_values_path = seg._array_sidecar_path(
            cache_root=cache_root,
            namespace="neuron_shapley_oracle",
            label=Path(image_path).stem,
            sidecar_key=sidecar_key,
            kind="oracle_values",
        )
        overlay_path = seg._array_sidecar_path(
            cache_root=cache_root,
            namespace="neuron_shapley_oracle",
            label=Path(image_path).stem,
            sidecar_key=sidecar_key,
            kind="oracle_overlay",
        )
        seg._save_array_sidecar(pool_indices_path, np.asarray(pool_indices, dtype=np.int64))
        seg._save_array_sidecar(oracle_values_path, np.asarray(shapley_values, dtype=np.float32))
        seg._save_array_sidecar(overlay_path, np.asarray(oracle_overlay, dtype=np.float32))
        delta_pool = delta[np.asarray(pool_indices, dtype=np.int64)]
        active_count = int(np.count_nonzero(delta > float(active_min_abs_delta)))
        return {
            "pool_size": int(len(pool_indices)),
            "pool_indices_path": str(pool_indices_path),
            "oracle_values_path": str(oracle_values_path),
            "oracle_overlay_path": str(overlay_path),
            "activation_shape": [int(v) for v in core_value["activation_shape"]],
            "oracle_imputer_kind": str(oracle_imputer_kind),
            "pool_selection_mode": str(pool_selection_mode),
            "unit_mode": str(unit_mode),
            "num_permutations": int(num_permutations),
            "active_min_abs_delta": float(active_min_abs_delta),
            "stratified_num_bins": int(stratified_num_bins),
            "active_candidate_count": active_count,
            "pool_delta_mean": float(np.mean(delta_pool)) if delta_pool.size else float("nan"),
            "pool_delta_min": float(np.min(delta_pool)) if delta_pool.size else float("nan"),
            "pool_delta_max": float(np.max(delta_pool)) if delta_pool.size else float("nan"),
            "oracle_value_mean": float(np.mean(shapley_values)) if shapley_values.size else float("nan"),
            "oracle_value_std": float(np.std(shapley_values)) if shapley_values.size else float("nan"),
        }
    finally:
        runner.close()
        seg._clear_all_backend_caches()


def _compute_classifier_evaluation_value(*, method_value, oracle_value, unit_mode, ndcg_k, recall_k, sign_threshold_ratio):
    unit_scores = _load_method_unit_scores(
        method_value,
        unit_mode=unit_mode,
        activation_shape=oracle_value.get("activation_shape"),
    )
    pool_indices = seg._load_array_sidecar(oracle_value["pool_indices_path"]).reshape(-1).astype(np.int64, copy=False)
    oracle_values = seg._load_array_sidecar(oracle_value["oracle_values_path"]).reshape(-1).astype(np.float64, copy=False)
    if unit_scores.size == 0 or pool_indices.size == 0:
        raise ValueError("Method scores or oracle pool is empty.")
    pool_scores = unit_scores[pool_indices]
    spearman = _spearman(pool_scores, oracle_values)
    ndcg = _ndcg_at_k(pool_scores, oracle_values, int(ndcg_k))
    recall = _recall_at_k(pool_scores, oracle_values, int(recall_k))
    sign_agreement = _sign_agreement(pool_scores, oracle_values, float(sign_threshold_ratio))
    return {
        "spearman": float(spearman),
        "ndcg_at_k": float(ndcg),
        "recall_at_k": float(recall),
        "sign_agreement": float(sign_agreement),
        "pool_size": int(pool_indices.size),
        "ndcg_k": int(ndcg_k),
        "recall_k": int(recall_k),
        "sign_threshold_ratio": float(sign_threshold_ratio),
        "method_abs_error": float(method_value.get("abs_error", float("nan"))),
        "selected_neurons": method_value.get("selected_neurons"),
        "score": float(spearman),
    }


class _ClassifierShapleyRunner:
    def __init__(self, *, image_path, layer_name, core_value, unit_mode):
        self.image_path = str(image_path)
        self.layer_name = str(layer_name)
        self.core_value = dict(core_value)
        self.unit_mode = aopc._normalize_unit_mode(unit_mode)
        self.x, self.image_np = IG.load_image(self.image_path)
        self.target_class = int(core_value["target_class"])
        self.clean_target_logit = float(core_value["clean_target_logit"])
        self.clean_act = seg._load_array_sidecar(core_value["clean_act_path"])
        self.black_act = seg._load_array_sidecar(core_value["black_act_path"])
        self.blur_act = seg._load_array_sidecar(core_value["blur_act_path"])
        self.activation_shape = tuple(int(v) for v in core_value["activation_shape"])
        self.clean_act_tensor = torch.from_numpy(self.clean_act).to(device=IG.DEVICE, dtype=IG.DTYPE)
        self.black_act_tensor = torch.from_numpy(self.black_act).to(device=IG.DEVICE, dtype=IG.DTYPE)
        self.blur_act_tensor = torch.from_numpy(self.blur_act).to(device=IG.DEVICE, dtype=IG.DTYPE)
        self._donor_cache = {
            "zero_baseline": torch.zeros_like(self.clean_act_tensor),
            "black_act": self.black_act_tensor,
            "blur_act": self.blur_act_tensor,
        }
        self._value_cache = {}
        self.modules = dict(IG.model.named_modules())

    def target_logit_with_masked_indices(self, masked_indices, *, donor_kind):
        masked_indices = tuple(sorted(int(v) for v in masked_indices))
        cache_key = (str(donor_kind), masked_indices)
        cached = self._value_cache.get(cache_key)
        if cached is not None:
            return cached
        if not masked_indices:
            return self.clean_target_logit
        donor = self._resolve_donor(masked_indices, donor_kind=donor_kind)
        index_tensor = torch.tensor(masked_indices, device=donor.device, dtype=torch.long)
        handle = self.modules[self.layer_name].register_forward_hook(
            lambda module, inp, out: _patch_units(out, donor, index_tensor, unit_mode=self.unit_mode)
        )
        try:
            with torch.no_grad():
                out = IG.model(self.x)
                _, logits = IG.split_classifier_output(out)
                value = float(logits[0, self.target_class].item())
        finally:
            handle.remove()
        self._value_cache[cache_key] = value
        return value

    def _resolve_donor(self, unit_indices, *, donor_kind):
        if donor_kind in self._donor_cache and donor_kind in {"zero_baseline", "black_act", "blur_act"}:
            return self._donor_cache[donor_kind]
        donor_cache_key = (str(donor_kind), tuple(int(v) for v in unit_indices))
        cached = self._donor_cache.get(donor_cache_key)
        if cached is not None:
            return cached
        if donor_kind == "layer_mean_exclusive":
            donor = _build_layer_mean_exclusive_unit_donor(
                self.clean_act_tensor,
                unit_indices,
                unit_mode=self.unit_mode,
            )
        elif donor_kind == "spatial_nli_same_channel":
            donor = _build_spatial_nli_same_channel_unit_donor(
                self.clean_act_tensor,
                unit_indices,
                unit_mode=self.unit_mode,
            )
        else:
            raise ValueError(f"Unsupported donor kind: {donor_kind}")
        self._donor_cache[donor_cache_key] = donor
        return donor

    def close(self):
        pass


def _compute_mc_shapley(*, runner, pool_indices, donor_kind, num_permutations, rng):
    pool_indices = np.asarray(pool_indices, dtype=np.int64)
    k = int(pool_indices.size)
    shapley = np.zeros(k, dtype=np.float64)
    empty_value = runner.target_logit_with_masked_indices(pool_indices.tolist(), donor_kind=donor_kind)
    for _ in range(int(num_permutations)):
        permutation = rng.permutation(k)
        kept_mask = np.zeros(k, dtype=bool)
        prev_value = empty_value
        for local_idx in permutation.tolist():
            kept_mask[local_idx] = True
            masked_global = pool_indices[~kept_mask]
            current_value = runner.target_logit_with_masked_indices(masked_global.tolist(), donor_kind=donor_kind)
            shapley[local_idx] += float(current_value - prev_value)
            prev_value = current_value
    shapley /= float(max(1, int(num_permutations)))
    return shapley.astype(np.float32, copy=False)


def _select_candidate_pool(*, delta, pool_size, mode, rng, active_min_abs_delta, stratified_num_bins):
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    all_indices = np.arange(delta.size, dtype=np.int64)
    active_indices = all_indices[delta > float(active_min_abs_delta)]
    if active_indices.size < int(pool_size):
        active_indices = all_indices
    if active_indices.size <= int(pool_size):
        return np.sort(active_indices.astype(np.int64, copy=False))

    if mode == "active_random":
        chosen = rng.choice(active_indices, size=int(pool_size), replace=False)
        return np.sort(np.asarray(chosen, dtype=np.int64))

    if mode != "stratified_activation_change":
        raise ValueError(f"Unsupported pool selection mode: {mode}")

    active_values = delta[active_indices]
    order = np.argsort(active_values)
    sorted_active = active_indices[order]
    bins = [list(chunk.astype(np.int64)) for chunk in np.array_split(sorted_active, int(stratified_num_bins)) if len(chunk)]
    chosen = []
    chosen_set = set()
    while len(chosen) < int(pool_size) and any(bins):
        for bucket in bins:
            if not bucket or len(chosen) >= int(pool_size):
                continue
            pick_pos = int(rng.integers(0, len(bucket)))
            value = int(bucket.pop(pick_pos))
            if value not in chosen_set:
                chosen.append(value)
                chosen_set.add(value)
    if len(chosen) < int(pool_size):
        remaining = [int(v) for v in active_indices.tolist() if int(v) not in chosen_set]
        extra = rng.choice(np.asarray(remaining, dtype=np.int64), size=int(pool_size) - len(chosen), replace=False)
        chosen.extend(int(v) for v in np.asarray(extra, dtype=np.int64).tolist())
    return np.sort(np.asarray(chosen[: int(pool_size)], dtype=np.int64))


def _patch_units(out, donor, index_tensor, *, unit_mode):
    return aopc._patch_single_tensor_units(out, donor, index_tensor, unit_mode=unit_mode)


def _reference_unit_scores(tensor, *, unit_mode):
    return aopc._unit_reference_scores_from_tensor(tensor, unit_mode=unit_mode).astype(np.float64, copy=False)


def _load_method_unit_scores(method_value, *, unit_mode, activation_shape=None):
    raw_scores = seg._load_array_sidecar(method_value["unit_scores_path"]).reshape(-1).astype(np.float64, copy=False)
    if unit_mode == "neuron":
        return raw_scores
    if activation_shape is None:
        activation_shape = method_value.get("activation_shape")
    activation_shape = tuple(int(v) for v in activation_shape) if activation_shape is not None else ()
    if not activation_shape:
        return raw_scores
    if len(activation_shape) < 4:
        return raw_scores
    _, channels, height, width = activation_shape
    expected = channels * height * width
    if raw_scores.size != expected:
        raise ValueError(
            f"Method score size mismatch for spatial_cell mode: got {raw_scores.size}, expected {expected}"
        )
    return raw_scores.reshape(channels, height * width).sum(axis=0).astype(np.float64, copy=False)


def _build_layer_mean_exclusive_unit_donor(clean_act, unit_indices, *, unit_mode):
    if unit_mode == "neuron":
        return seg._build_layer_mean_exclusive_donor(clean_act, unit_indices)
    donor = clean_act.clone()
    if donor.ndim <= 2:
        return donor
    flat = donor.reshape(donor.shape[0], donor.shape[1], -1)
    clean_flat = clean_act.reshape(clean_act.shape[0], clean_act.shape[1], -1)
    mask = torch.zeros(flat.shape[2], device=flat.device, dtype=torch.bool)
    if unit_indices:
        mask[torch.tensor(unit_indices, device=flat.device, dtype=torch.long)] = True
    if torch.any(~mask):
        mean_value = clean_flat[:, :, ~mask].mean()
    else:
        mean_value = clean_flat.mean()
    flat[:, :, mask] = mean_value
    return flat.reshape_as(donor)


def _build_spatial_nli_same_channel_unit_donor(clean_act, unit_indices, *, unit_mode):
    if unit_mode == "neuron":
        return seg._build_spatial_nli_same_channel_donor(clean_act, unit_indices)
    if clean_act.ndim != 4:
        raise ValueError(
            "spatial_nli_same_channel donor requires activation shape [B, C, H, W], "
            f"got {tuple(clean_act.shape)}"
        )
    if clean_act.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 for donor construction, got shape={tuple(clean_act.shape)}")

    donor_np = clean_act.detach().cpu().numpy().astype(np.float32, copy=True)
    _, channels, height, width = donor_np.shape
    cell_mask = np.zeros(height * width, dtype=bool)
    if unit_indices:
        cell_mask[np.asarray(unit_indices, dtype=np.int64)] = True
    cell_mask = cell_mask.reshape(height, width)

    for channel_idx in range(channels):
        if not np.any(cell_mask):
            continue
        channel_values = donor_np[0, channel_idx]
        solved = seg._impute_single_channel(channel_values, cell_mask)
        if solved.shape != channel_values.shape:
            fallback = _build_layer_mean_exclusive_unit_donor(clean_act, unit_indices, unit_mode=unit_mode)
            return fallback
        donor_np[0, channel_idx, cell_mask] = solved[cell_mask]
    return torch.from_numpy(donor_np).to(device=clean_act.device, dtype=clean_act.dtype)


def _spearman(scores, oracle_values):
    scores = np.asarray(scores, dtype=np.float64)
    oracle_values = np.asarray(oracle_values, dtype=np.float64)
    if scores.size == 0 or oracle_values.size == 0:
        return float("nan")
    if np.allclose(scores, scores[0]) or np.allclose(oracle_values, oracle_values[0]):
        return float("nan")
    value = spearmanr(scores, oracle_values).statistic
    return float(value) if value == value else float("nan")


def _ndcg_at_k(scores, oracle_values, k):
    scores = np.asarray(scores, dtype=np.float64)
    oracle_values = np.asarray(oracle_values, dtype=np.float64)
    if scores.size == 0 or oracle_values.size == 0:
        return float("nan")
    k = int(max(1, min(int(k), scores.size)))
    relevance = np.maximum(oracle_values, 0.0)
    ideal_order = np.argsort(-relevance)
    ideal = _dcg(relevance[ideal_order][:k])
    if ideal <= 0.0:
        return float("nan")
    pred_order = np.argsort(-scores)
    actual = _dcg(relevance[pred_order][:k])
    return float(actual / ideal)


def _recall_at_k(scores, oracle_values, k):
    scores = np.asarray(scores, dtype=np.float64)
    oracle_values = np.asarray(oracle_values, dtype=np.float64)
    if scores.size == 0 or oracle_values.size == 0:
        return float("nan")
    k = int(max(1, min(int(k), scores.size)))
    oracle_positive = np.maximum(oracle_values, 0.0)
    if np.allclose(oracle_positive, 0.0):
        return float("nan")
    oracle_order = np.argsort(-oracle_positive)[:k]
    pred_order = np.argsort(-scores)[:k]
    oracle_set = set(int(v) for v in oracle_order.tolist())
    pred_set = set(int(v) for v in pred_order.tolist())
    return float(len(oracle_set & pred_set) / max(1, len(oracle_set)))


def _sign_agreement(scores, oracle_values, threshold_ratio):
    scores = np.asarray(scores, dtype=np.float64)
    oracle_values = np.asarray(oracle_values, dtype=np.float64)
    if scores.size == 0 or oracle_values.size == 0:
        return float("nan")
    max_abs = float(np.max(np.abs(oracle_values)))
    if max_abs <= 0.0:
        return float("nan")
    mask = np.abs(oracle_values) >= float(threshold_ratio) * max_abs
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.sign(scores[mask]) == np.sign(oracle_values[mask])))


def _dcg(relevance):
    relevance = np.asarray(relevance, dtype=np.float64)
    if relevance.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, relevance.size + 2, dtype=np.float64))
    return float(np.sum(relevance / discounts))


def _build_oracle_overlay(*, shapley_values, pool_indices, activation_shape, image_shape, unit_mode):
    _, channels, height, width = activation_shape
    if unit_mode == "neuron":
        tensor = np.zeros((channels * height * width,), dtype=np.float32)
        tensor[np.asarray(pool_indices, dtype=np.int64)] = np.asarray(shapley_values, dtype=np.float32)
        tensor = tensor.reshape(channels, height, width)
        overlay = tensor.sum(axis=0).astype(np.float32, copy=False)
    else:
        overlay = np.zeros((height * width,), dtype=np.float32)
        overlay[np.asarray(pool_indices, dtype=np.int64)] = np.asarray(shapley_values, dtype=np.float32)
        overlay = overlay.reshape(height, width)
    if tuple(overlay.shape) != tuple(image_shape):
        overlay = IG._resize_map_nearest(overlay, image_shape).astype(np.float32, copy=False)
    return overlay


def _build_summary(
    *,
    rows,
    core_rows,
    oracle_rows,
    method_rows,
    method_specs,
    layer_name,
    unit_mode,
    n_steps,
    pool_size,
    num_permutations,
    pool_selection_mode,
    active_min_abs_delta,
    stratified_num_bins,
    oracle_imputer_kind,
    random_seed,
    ndcg_k,
    recall_k,
    sign_threshold_ratio,
    cache_root,
    preview_image_paths,
):
    method_names = [str(spec["name"]) for spec in method_specs]
    metric_names = ("spearman", "ndcg_at_k", "recall_at_k", "sign_agreement")
    summary_rows = []
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method_name"], []).append(row)
    for method_name in method_names:
        subset = by_method.get(method_name, [])
        summary_rows.append(
            {
                "method_name": method_name,
                "spearman_mean": float(seg._stats_record([row["spearman"] for row in subset])["mean"]),
                "spearman_std": float(seg._stats_record([row["spearman"] for row in subset])["std"]),
                "ndcg_at_k_mean": float(seg._stats_record([row["ndcg_at_k"] for row in subset])["mean"]),
                "ndcg_at_k_std": float(seg._stats_record([row["ndcg_at_k"] for row in subset])["std"]),
                "recall_at_k_mean": float(seg._stats_record([row["recall_at_k"] for row in subset])["mean"]),
                "recall_at_k_std": float(seg._stats_record([row["recall_at_k"] for row in subset])["std"]),
                "sign_agreement_mean": float(seg._stats_record([row["sign_agreement"] for row in subset])["mean"]),
                "sign_agreement_std": float(seg._stats_record([row["sign_agreement"] for row in subset])["std"]),
                "runtime_s_mean": float(seg._stats_record([row["method_duration_s"] for row in subset])["mean"]),
                "benchmark_runtime_s_mean": float(seg._stats_record([row["benchmark_duration_s"] for row in subset])["mean"]),
                "method_abs_error_mean": float(seg._stats_record([row["method_abs_error"] for row in subset])["mean"]),
                "selected_neurons_mean": float(seg._stats_record([row.get("selected_neurons") for row in subset])["mean"]),
                "n_images": int(len(subset)),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            -float(row["spearman_mean"]) if row["spearman_mean"] == row["spearman_mean"] else float("inf"),
            -float(row["ndcg_at_k_mean"]) if row["ndcg_at_k_mean"] == row["ndcg_at_k_mean"] else float("inf"),
            method_names.index(row["method_name"]),
        )
    )
    best_method = summary_rows[0]["method_name"] if summary_rows else None

    pairwise = {}
    for left_name in method_names:
        pairwise[left_name] = {}
        left_subset = {row["image_name"]: row for row in by_method.get(left_name, [])}
        for right_name in method_names:
            right_subset = {row["image_name"]: row for row in by_method.get(right_name, [])}
            common = sorted(set(left_subset) & set(right_subset))
            if not common:
                pairwise[left_name][right_name] = float("nan")
                continue
            wins = []
            for image_name in common:
                left_score = float(left_subset[image_name]["spearman"])
                right_score = float(right_subset[image_name]["spearman"])
                wins.append(1.0 if left_score > right_score else 0.0)
            pairwise[left_name][right_name] = float(np.mean(np.asarray(wins, dtype=np.float64)))

    return {
        "task": "classifier",
        "layer_name": str(layer_name),
        "unit_mode": str(unit_mode),
        "n_steps": int(n_steps),
        "pool_size": int(pool_size),
        "num_permutations": int(num_permutations),
        "pool_selection_mode": str(pool_selection_mode),
        "active_min_abs_delta": float(active_min_abs_delta),
        "stratified_num_bins": int(stratified_num_bins),
        "oracle_imputer_kind": str(oracle_imputer_kind),
        "random_seed": int(random_seed),
        "ndcg_k": int(ndcg_k),
        "recall_k": int(recall_k),
        "sign_threshold_ratio": float(sign_threshold_ratio),
        "metric_names": list(metric_names),
        "method_names": method_names,
        "best_method": best_method,
        "summary_rows": summary_rows,
        "pairwise_win_rates": pairwise,
        "n_images": int(len(core_rows)),
        "cache_root": str(cache_root),
        "preview_image_paths": [str(path) for path in preview_image_paths],
        "core_summary": {
            "n_units_total": seg._stats_record([row["core"]["n_units_total"] for row in core_rows]),
            "core_runtime_s": seg._stats_record([row["core_duration_s"] for row in core_rows]),
            "oracle_runtime_s": seg._stats_record([row["oracle_duration_s"] for row in oracle_rows]),
            "active_candidate_count": seg._stats_record([row["active_candidate_count"] for row in oracle_rows]),
            "pool_delta_mean": seg._stats_record([row["pool_delta_mean"] for row in oracle_rows]),
        },
    }


def _render_and_save_report_figures(*, run_dir, summary, rows):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "summary_metrics": seg._save_figure(
            _plot_summary_metrics(summary),
            figure_dir / "summary_metrics.png",
        ),
        "distribution_metrics": seg._save_figure(
            _plot_distribution_metrics(summary, rows),
            figure_dir / "distribution_metrics.png",
        ),
        "pairwise_spearman": seg._save_figure(
            _plot_pairwise_spearman(summary),
            figure_dir / "pairwise_spearman.png",
        ),
    }
    return figures


def _render_preview_sections(*, run_dir, summary, oracle_rows, method_rows):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    oracle_index = {row["image_path"]: row for row in oracle_rows}
    method_index = {(row["image_path"], row["method_name"]): row for row in method_rows}
    sections = []
    for page_idx, image_chunk in enumerate(seg._chunked(summary["preview_image_paths"], DEFAULT_PREVIEW_IMAGES), start=1):
        fig = _build_preview_figure(
            image_paths=image_chunk,
            method_names=summary["method_names"],
            oracle_index=oracle_index,
            method_index=method_index,
        )
        figure_path = figure_dir / f"preview_page_{page_idx}.png"
        saved_path = seg._save_figure(fig, figure_path)
        sections.append(
            {
                "page_idx": int(page_idx),
                "figure_path": saved_path,
                "method_names": list(summary["method_names"]),
            }
        )
    return sections


def _plot_summary_metrics(summary):
    metrics = [
        ("spearman", "Spearman"),
        ("ndcg_at_k", f"NDCG@{summary['ndcg_k']}"),
        ("recall_at_k", f"Recall@{summary['recall_k']}"),
        ("sign_agreement", "Sign Agreement"),
    ]
    rows = summary["summary_rows"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.5 * len(metrics), max(4.8, 0.48 * len(rows) + 2.0)), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]
    labels = [row["method_name"] for row in rows]
    y = np.arange(len(rows))
    colors = [plt.get_cmap("tab10")(idx % 10) for idx in range(len(rows))]
    for ax, (metric_key, title) in zip(axes, metrics):
        values = np.asarray([row[f"{metric_key}_mean"] for row in rows], dtype=np.float64)
        stds = np.asarray([row[f"{metric_key}_std"] for row in rows], dtype=np.float64)
        ax.errorbar(values, y, xerr=stds, fmt="o", ecolor="black", color="black", capsize=4)
        ax.scatter(values, y, c=colors, s=45, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax is axes[0] else [""] * len(labels))
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Monte-Carlo Shapley audit summary", fontsize=14)
    return fig


def _plot_distribution_metrics(summary, rows):
    metrics = [
        ("spearman", "Spearman"),
        ("ndcg_at_k", f"NDCG@{summary['ndcg_k']}"),
        ("recall_at_k", f"Recall@{summary['recall_k']}"),
        ("sign_agreement", "Sign Agreement"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, (metric_key, title) in zip(axes, metrics):
        data = []
        labels = []
        for method_name in summary["method_names"]:
            subset = [float(row[metric_key]) for row in rows if row["method_name"] == method_name and float(row[metric_key]) == float(row[metric_key])]
            if not subset:
                continue
            data.append(subset)
            labels.append(method_name)
        if data:
            ax.boxplot(data, vert=False, labels=labels, patch_artist=True)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)
    return fig


def _plot_pairwise_spearman(summary):
    method_names = summary["method_names"]
    matrix = np.full((len(method_names), len(method_names)), np.nan, dtype=np.float64)
    for row_idx, left_name in enumerate(method_names):
        for col_idx, right_name in enumerate(method_names):
            matrix[row_idx, col_idx] = float(summary["pairwise_win_rates"][left_name][right_name])
    fig, ax = plt.subplots(
        figsize=(2.2 + 0.8 * len(method_names), 2.2 + 0.6 * len(method_names)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_yticklabels(method_names)
    ax.set_title("Pairwise win-rate by Spearman")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(col_idx, row_idx, "n/a" if value != value else f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(row > col)")
    return fig


def _build_preview_figure(*, image_paths, method_names, oracle_index, method_index):
    n_rows = len(image_paths)
    n_cols = 2 + len(method_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.4 * n_rows + 0.6),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle("Neuron Shapley benchmark previews", fontsize=14)
    for row_idx, image_path in enumerate(image_paths):
        _, image_np = IG.load_image(str(image_path))
        raw_ax = axes[row_idx, 0]
        raw_ax.imshow(image_np, interpolation="nearest")
        raw_ax.axis("off")
        if row_idx == 0:
            raw_ax.set_title("Raw", fontsize=9)
        raw_ax.text(-0.02, 0.5, Path(image_path).name, transform=raw_ax.transAxes, va="center", ha="right", fontsize=8, rotation=90)

        oracle_ax = axes[row_idx, 1]
        oracle = oracle_index.get(str(image_path))
        if oracle is not None:
            overlay = seg._load_array_sidecar(oracle["oracle_overlay_path"])
            overlay = seg._normalize_map(overlay)
            if tuple(overlay.shape) != tuple(image_np.shape[:2]):
                overlay = IG._resize_map_nearest(overlay, image_np.shape[:2]).astype(np.float32, copy=False)
            oracle_ax.imshow(image_np, interpolation="nearest")
            oracle_ax.imshow(overlay, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
            oracle_ax.axis("off")
        else:
            oracle_ax.set_axis_off()
        if row_idx == 0:
            oracle_ax.set_title("Oracle", fontsize=9)

        for col_idx, method_name in enumerate(method_names, start=2):
            ax = axes[row_idx, col_idx]
            record = method_index.get((str(image_path), method_name))
            if record is None:
                ax.set_axis_off()
                continue
            overlay = seg._load_array_sidecar(record["overlay_map_path"])
            overlay = seg._normalize_map(overlay)
            if tuple(overlay.shape) != tuple(image_np.shape[:2]):
                overlay = IG._resize_map_nearest(overlay, image_np.shape[:2]).astype(np.float32, copy=False)
            ax.imshow(image_np, interpolation="nearest")
            ax.imshow(overlay, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(method_name, fontsize=9)
    return fig


def _build_report_markdown(summary, *, figures, preview_sections):
    lines = [
        "# Neuron Shapley Audit Benchmark",
        "",
        "Classifier-only Monte-Carlo Shapley audit benchmark on real model and dataset.",
        "",
        "## Configuration",
        "",
        f"- layer_name=`{summary['layer_name']}`",
        f"- unit_mode=`{summary['unit_mode']}`",
        f"- n_steps=`{summary['n_steps']}`",
        f"- pool_size=`{summary['pool_size']}`",
        f"- num_permutations=`{summary['num_permutations']}`",
        f"- pool_selection_mode=`{summary['pool_selection_mode']}`",
        f"- active_min_abs_delta=`{summary['active_min_abs_delta']}`",
        f"- stratified_num_bins=`{summary['stratified_num_bins']}`",
        f"- oracle_imputer_kind=`{summary['oracle_imputer_kind']}`",
        f"- random_seed=`{summary['random_seed']}`",
        f"- n_images=`{summary['n_images']}`",
        "",
        "## Summary Table",
        "",
        _build_summary_table(summary["summary_rows"], summary=summary),
        "",
    ]
    for key in ("summary_metrics", "distribution_metrics", "pairwise_spearman"):
        path = figures.get(key)
        if path:
            lines.extend([f"![]({seg._relative_markdown_path(path)})", ""])
    if preview_sections:
        lines.extend(["## Visual Preview", ""])
        for section in preview_sections:
            lines.append(f"Page {section['page_idx']} | methods: `{section['method_names']}`")
            lines.append("")
            lines.append(f"![]({seg._relative_markdown_path(section['figure_path'])})")
            lines.append("")
    return "\n".join(lines)


def _build_summary_table(rows, *, summary):
    lines = [
        "| Method | Spearman | NDCG@k | Recall@k | Sign agreement | runtime_s | benchmark_runtime_s | abs_error |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {spearman} | {ndcg} | {recall} | {sign} | {runtime} | {bench} | {abs_error} |".format(
                method=row["method_name"],
                spearman=seg._format_number(row["spearman_mean"]),
                ndcg=seg._format_number(row["ndcg_at_k_mean"]),
                recall=seg._format_number(row["recall_at_k_mean"]),
                sign=seg._format_number(row["sign_agreement_mean"]),
                runtime=seg._format_number(row["runtime_s_mean"]),
                bench=seg._format_number(row["benchmark_runtime_s_mean"]),
                abs_error=seg._format_number(row["method_abs_error_mean"]),
            )
        )
    return "\n".join(lines)


def _default_method_name(spec):
    kind = str(spec.get("kind", "method"))
    if kind == "ig":
        return "IG"
    if kind == "naa":
        return "NAA"
    selection_top_k = int(spec.get("selection_top_k", 8000))
    segment_start = float(spec.get("segment_start", 0.0))
    segment_end = float(spec.get("segment_end", 0.1))
    return f"Cheap-IG+[{segment_start:g},{segment_end:g}]/k{selection_top_k}/{spec.get('fill_mode', 'zero')}"


def _image_seed(base_seed, image_index, image_path):
    digest = seg._config_hash({"seed": int(base_seed), "image_index": int(image_index), "image_path": str(image_path)})
    return int(digest[:8], 16)


def _require_scipy_stats():
    if spearmanr is None or rankdata is None:
        raise RuntimeError(
            "Neuron Shapley benchmark requires scipy.stats, but import failed: "
            f"{SCIPY_STATS_IMPORT_ERROR}"
        )
