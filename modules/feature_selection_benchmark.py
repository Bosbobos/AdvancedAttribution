from __future__ import annotations

"""Feature-selection benchmark inspired by How Important Is a Neuron? Sec. 5.2."""

from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modules import alpha_segment_benchmark as seg


DEFAULT_CACHE_ROOT = "output/feature_selection_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_FILENAME = "feature_selection_report.md"
DEFAULT_SUMMARY_JSON = "feature_selection_summary.json"
DEFAULT_LAYER_NAME = "model.6"
DEFAULT_N_STEPS = 128
DEFAULT_TRAIN_PER_CLASS = 10
DEFAULT_EVAL_PER_CLASS = 10
DEFAULT_K_VALUES = (4, 8, 16, 32, 64)
DEFAULT_RANDOM_SEED = 0
DEFAULT_CHANNEL_AGGREGATION = "positive_mean"
DEFAULT_SELECTION_RULE = "max_over_classes"


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


def default_feature_selection_tasks(image_paths, *, min_images_per_class=20):
    by_label = _group_images_by_label(image_paths)
    counts = {label: len(paths) for label, paths in by_label.items()}
    eligible_cats = [label for label in sorted(by_label) if label[:1].isupper() and counts[label] >= int(min_images_per_class)]
    eligible_dogs = [label for label in sorted(by_label) if label[:1].islower() and counts[label] >= int(min_images_per_class)]

    tasks = []
    if len(eligible_cats) >= 5:
        tasks.append({"name": "cats_a", "classes": eligible_cats[:5]})
    if len(eligible_cats) >= 10:
        tasks.append({"name": "cats_b", "classes": eligible_cats[5:10]})
    if len(eligible_dogs) >= 5:
        tasks.append({"name": "dogs_a", "classes": eligible_dogs[:5]})
    if len(eligible_dogs) >= 10:
        tasks.append({"name": "dogs_b", "classes": eligible_dogs[5:10]})

    if tasks:
        return tasks

    eligible_all = [label for label in sorted(by_label) if counts[label] >= int(min_images_per_class)]
    for idx, chunk in enumerate(seg._chunked(eligible_all, 5), start=1):
        if len(chunk) == 5:
            tasks.append({"name": f"task_{idx}", "classes": list(chunk)})
    if not tasks:
        raise ValueError("Could not build any 5-way feature-selection tasks from the provided image_paths.")
    return tasks


def benchmark_classifier_feature_selection(
    *,
    image_paths,
    method_specs,
    class_tasks=None,
    layer_name=DEFAULT_LAYER_NAME,
    n_steps=DEFAULT_N_STEPS,
    train_per_class=DEFAULT_TRAIN_PER_CLASS,
    eval_per_class=DEFAULT_EVAL_PER_CLASS,
    k_values=DEFAULT_K_VALUES,
    channel_aggregation=DEFAULT_CHANNEL_AGGREGATION,
    selection_rule=DEFAULT_SELECTION_RULE,
    random_seed=DEFAULT_RANDOM_SEED,
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
):
    image_paths = [str(Path(path)) for path in image_paths]
    normalized_method_specs = seg._normalize_method_specs(method_specs)
    k_values = sorted({int(k) for k in k_values if int(k) > 0})
    if not k_values:
        raise ValueError("k_values must contain at least one positive integer.")
    if channel_aggregation not in {"positive_mean", "signed_mean"}:
        raise ValueError("channel_aggregation must be `positive_mean` or `signed_mean`.")
    if selection_rule not in {"max_over_classes"}:
        raise ValueError("selection_rule must be `max_over_classes`.")

    by_label = _group_images_by_label(image_paths)
    if class_tasks is None:
        class_tasks = default_feature_selection_tasks(image_paths)
    normalized_tasks = _normalize_tasks(
        class_tasks,
        by_label=by_label,
        train_per_class=int(train_per_class),
        eval_per_class=int(eval_per_class),
        random_seed=int(random_seed),
    )

    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = seg._prepare_existing_output_dir(target_dir)
        else:
            run_name = (
                f"feature_selection_{seg._safe_slug(layer_name)}"
                f"_tasks{len(normalized_tasks)}_k{seg._safe_slug('-'.join(str(v) for v in k_values))}"
            )
            run_dir = seg._prepare_output_dir(output_dir, run_name)

    used_images = sorted({path for task in normalized_tasks for path in task["train_images"] + task["eval_images"]})
    core_rows = []
    method_rows = []
    core_by_image = {}
    method_by_image = {}
    for image_path in used_images:
        label = _label_from_path(image_path)
        core_record = seg._load_core_record(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            blur_sigma=seg.DEFAULT_BLUR_SIGMA,
            budget_percentiles=[1, 20],
            cache_root=cache_root,
            refresh=refresh_core,
        )
        if core_record.get("error") is not None or core_record.get("value") is None:
            raise RuntimeError(
                f"Failed to build feature-selection core for {image_path}: {core_record.get('error')} "
                f"(cache={core_record.get('cache_path')})"
            )
        core_value = core_record["value"]
        gap_features = _gap_features_from_core(core_value)
        core_row = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "label": label,
            "core_cache_path": core_record["cache_path"],
            "core_from_cache": core_record["from_cache"],
            "core_duration_s": core_record["duration_s"],
            "activation_shape": [int(v) for v in core_value["activation_shape"]],
            "clean_act_path": core_value["clean_act_path"],
            "gap_features": gap_features.tolist(),
        }
        core_rows.append(core_row)
        core_by_image[image_path] = {
            "row": core_row,
            "gap_features": gap_features,
            "activation_shape": tuple(int(v) for v in core_value["activation_shape"]),
        }

        method_by_image[image_path] = {}
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
            channel_scores = _channel_scores_from_method(
                method_value=method_value,
                activation_shape=core_by_image[image_path]["activation_shape"],
            )
            method_row = {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "label": label,
                "method_name": method_spec["name"],
                "method_id": method_spec["id"],
                "kind": method_spec["kind"],
                "method_cache_path": method_record["cache_path"],
                "method_from_cache": method_record["from_cache"],
                "method_duration_s": method_record["duration_s"],
                "unit_scores_path": method_value["unit_scores_path"],
                "abs_error": float(method_value.get("abs_error", float("nan"))),
                "selected_neurons": method_value.get("selected_neurons"),
                "channel_scores": channel_scores.tolist(),
            }
            method_rows.append(method_row)
            method_by_image[image_path][method_spec["name"]] = {
                "row": method_row,
                "channel_scores": channel_scores,
            }

    rows = []
    for task in normalized_tasks:
        task_classes = list(task["classes"])
        class_to_train = {label: list(paths) for label, paths in task["train_by_class"].items()}
        class_to_eval = {label: list(paths) for label, paths in task["eval_by_class"].items()}
        class_names = list(task_classes)
        label_to_index = {label: idx for idx, label in enumerate(class_names)}

        for method_spec in normalized_method_specs:
            per_class_scores = {}
            for label in class_names:
                vectors = [
                    method_by_image[path][method_spec["name"]]["channel_scores"]
                    for path in class_to_train[label]
                ]
                per_class_scores[label] = _aggregate_class_channel_scores(
                    vectors,
                    aggregation_mode=channel_aggregation,
                )
            global_scores = _select_global_channel_scores(
                per_class_scores,
                selection_rule=selection_rule,
            )

            train_features = {
                path: core_by_image[path]["gap_features"]
                for label in class_names
                for path in class_to_train[label]
            }
            eval_features = {
                path: core_by_image[path]["gap_features"]
                for label in class_names
                for path in class_to_eval[label]
            }

            for k in k_values:
                selected_channels = _topk_indices(global_scores, k)
                x_train = np.stack([train_features[path][selected_channels] for label in class_names for path in class_to_train[label]], axis=0)
                y_train = np.asarray([label_to_index[label] for label in class_names for path in class_to_train[label]], dtype=np.int64)
                x_eval = np.stack([eval_features[path][selected_channels] for label in class_names for path in class_to_eval[label]], axis=0)
                y_eval = np.asarray([label_to_index[label] for label in class_names for path in class_to_eval[label]], dtype=np.int64)
                metrics = _train_and_evaluate_linear_probe(
                    x_train=x_train,
                    y_train=y_train,
                    x_eval=x_eval,
                    y_eval=y_eval,
                    random_seed=int(random_seed),
                )
                rows.append(
                    {
                        "task_name": task["name"],
                        "task_classes": class_names,
                        "method_name": method_spec["name"],
                        "method_id": method_spec["id"],
                        "k": int(k),
                        "n_train": int(x_train.shape[0]),
                        "n_eval": int(x_eval.shape[0]),
                        "accuracy": float(metrics["accuracy"]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "selected_channels": [int(v) for v in selected_channels.tolist()],
                        "selected_channel_scores_head": [float(v) for v in global_scores[selected_channels[: min(10, selected_channels.size)]].tolist()],
                    }
                )

    summary = _build_summary(
        rows=rows,
        core_rows=core_rows,
        method_rows=method_rows,
        method_specs=normalized_method_specs,
        tasks=normalized_tasks,
        layer_name=layer_name,
        n_steps=n_steps,
        k_values=k_values,
        train_per_class=int(train_per_class),
        eval_per_class=int(eval_per_class),
        channel_aggregation=channel_aggregation,
        selection_rule=selection_rule,
        random_seed=int(random_seed),
        cache_root=cache_root,
    )

    figures = {}
    report_md = _build_report_markdown(summary, figures={})
    report_path = None
    summary_path = None
    if save_output and run_dir is not None:
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
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "k_values": [int(v) for v in k_values],
        "train_per_class": int(train_per_class),
        "eval_per_class": int(eval_per_class),
        "channel_aggregation": str(channel_aggregation),
        "selection_rule": str(selection_rule),
        "random_seed": int(random_seed),
        "method_specs": normalized_method_specs,
        "tasks": normalized_tasks,
        "rows": rows,
        "core_rows": core_rows,
        "method_rows": method_rows,
        "summary": summary,
        "report_markdown": report_md,
        "report_path": str(report_path) if report_path is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "figures": figures,
        "output_dir": str(run_dir) if run_dir is not None else None,
        "cache_root": str(cache_root),
    }


def render_feature_selection_report(result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = _render_and_save_report_figures(
        run_dir=output_dir,
        summary=result["summary"],
        rows=result["rows"],
    )
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


def _label_from_path(path):
    return Path(path).stem.rsplit("_", 1)[0]


def _group_images_by_label(image_paths):
    by_label = defaultdict(list)
    for image_path in image_paths:
        by_label[_label_from_path(image_path)].append(str(image_path))
    for label in list(by_label):
        by_label[label] = sorted(by_label[label], key=lambda path: Path(path).name.lower())
    return dict(by_label)


def _normalize_tasks(class_tasks, *, by_label, train_per_class, eval_per_class, random_seed):
    tasks = []
    for task_idx, task in enumerate(class_tasks):
        name = str(task["name"])
        classes = [str(label) for label in task["classes"]]
        if len(classes) != len(set(classes)):
            raise ValueError(f"Duplicate class in task {name}: {classes}")
        train_by_class = {}
        eval_by_class = {}
        for class_idx, label in enumerate(classes):
            paths = list(by_label.get(label, []))
            required = int(train_per_class) + int(eval_per_class)
            if len(paths) < required:
                raise ValueError(
                    f"Task {name} requires {required} images for class {label}, got {len(paths)}."
                )
            rng = np.random.default_rng(int(random_seed) + 1000 * task_idx + class_idx)
            order = rng.permutation(len(paths))
            ordered_paths = [paths[int(idx)] for idx in order.tolist()]
            train_by_class[label] = ordered_paths[: int(train_per_class)]
            eval_by_class[label] = ordered_paths[int(train_per_class) : int(train_per_class) + int(eval_per_class)]
        tasks.append(
            {
                "name": name,
                "classes": classes,
                "train_by_class": train_by_class,
                "eval_by_class": eval_by_class,
                "train_images": [path for label in classes for path in train_by_class[label]],
                "eval_images": [path for label in classes for path in eval_by_class[label]],
            }
        )
    return tasks


def _gap_features_from_core(core_value):
    clean_act = seg._load_array_sidecar(core_value["clean_act_path"]).astype(np.float32, copy=False)
    if clean_act.ndim != 4 or clean_act.shape[0] != 1:
        raise ValueError(f"Expected clean activation with shape [1,C,H,W], got {tuple(clean_act.shape)}")
    return clean_act[0].reshape(clean_act.shape[1], -1).mean(axis=1).astype(np.float32, copy=False)


def _channel_scores_from_method(*, method_value, activation_shape):
    raw_scores = seg._load_array_sidecar(method_value["unit_scores_path"]).reshape(-1).astype(np.float64, copy=False)
    activation_shape = tuple(int(v) for v in activation_shape)
    if len(activation_shape) < 4:
        raise ValueError(f"Expected activation shape [1,C,H,W], got {activation_shape}")
    _, channels, height, width = activation_shape
    expected = channels * height * width
    if raw_scores.size != expected:
        raise ValueError(
            f"Method score size mismatch: got {raw_scores.size}, expected {expected} for shape {activation_shape}"
        )
    return raw_scores.reshape(channels, height * width).sum(axis=1).astype(np.float32, copy=False)


def _aggregate_class_channel_scores(vectors, *, aggregation_mode):
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in vectors], axis=0)
    if aggregation_mode == "positive_mean":
        stacked = np.maximum(stacked, 0.0)
    return stacked.mean(axis=0).astype(np.float32, copy=False)


def _select_global_channel_scores(per_class_scores, *, selection_rule):
    stacked = np.stack([np.asarray(v, dtype=np.float64) for v in per_class_scores.values()], axis=0)
    if selection_rule == "max_over_classes":
        return stacked.max(axis=0).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported selection_rule: {selection_rule}")


def _topk_indices(scores, k):
    order = seg._stable_descending_order(scores)
    k = int(max(1, min(int(k), order.size)))
    return order[:k].astype(np.int64, copy=False)


def _train_and_evaluate_linear_probe(*, x_train, y_train, x_eval, y_eval, random_seed):
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=int(random_seed),
            solver="lbfgs",
        ),
    )
    probe.fit(x_train, y_train)
    pred = probe.predict(x_eval)
    return {
        "accuracy": float(accuracy_score(y_eval, pred)),
        "macro_f1": float(f1_score(y_eval, pred, average="macro")),
    }


def _build_summary(
    *,
    rows,
    core_rows,
    method_rows,
    method_specs,
    tasks,
    layer_name,
    n_steps,
    k_values,
    train_per_class,
    eval_per_class,
    channel_aggregation,
    selection_rule,
    random_seed,
    cache_root,
):
    method_names = [str(spec["name"]) for spec in method_specs]
    summary_rows = []
    for method_name in method_names:
        for k in k_values:
            subset = [row for row in rows if row["method_name"] == method_name and int(row["k"]) == int(k)]
            summary_rows.append(
                {
                    "method_name": method_name,
                    "k": int(k),
                    "accuracy_mean": float(seg._stats_record([row["accuracy"] for row in subset])["mean"]),
                    "accuracy_std": float(seg._stats_record([row["accuracy"] for row in subset])["std"]),
                    "macro_f1_mean": float(seg._stats_record([row["macro_f1"] for row in subset])["mean"]),
                    "macro_f1_std": float(seg._stats_record([row["macro_f1"] for row in subset])["std"]),
                    "n_tasks": int(len(subset)),
                }
            )
    best_rows = []
    for method_name in method_names:
        method_summary = [row for row in summary_rows if row["method_name"] == method_name]
        method_summary.sort(key=lambda row: (-float(row["accuracy_mean"]), -float(row["macro_f1_mean"]), int(row["k"])))
        if method_summary:
            best_rows.append(
                {
                    "method_name": method_name,
                    "best_k": int(method_summary[0]["k"]),
                    "best_accuracy_mean": float(method_summary[0]["accuracy_mean"]),
                    "best_macro_f1_mean": float(method_summary[0]["macro_f1_mean"]),
                }
            )

    reference_k = min(k_values, key=lambda value: (abs(int(value) - 16), int(value)))
    overlap_by_task = {}
    for task in tasks:
        task_rows = [row for row in rows if row["task_name"] == task["name"] and int(row["k"]) == int(reference_k)]
        overlap_by_task[task["name"]] = _selected_overlap_matrix(task_rows, method_names)

    return {
        "task": "classifier",
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "k_values": [int(v) for v in k_values],
        "train_per_class": int(train_per_class),
        "eval_per_class": int(eval_per_class),
        "channel_aggregation": str(channel_aggregation),
        "selection_rule": str(selection_rule),
        "random_seed": int(random_seed),
        "method_names": method_names,
        "tasks": [{"name": task["name"], "classes": list(task["classes"])} for task in tasks],
        "summary_rows": summary_rows,
        "best_rows": best_rows,
        "reference_k_for_overlap": int(reference_k),
        "selected_overlap": overlap_by_task,
        "core_summary": {
            "n_images": int(len(core_rows)),
            "n_methods": int(len(method_specs)),
            "core_runtime_s": seg._stats_record([row["core_duration_s"] for row in core_rows]),
            "method_runtime_s": seg._stats_record([row["method_duration_s"] for row in method_rows]),
        },
        "cache_root": str(cache_root),
    }


def _selected_overlap_matrix(task_rows, method_names):
    matrix = {left: {} for left in method_names}
    by_method = {row["method_name"]: set(int(v) for v in row["selected_channels"]) for row in task_rows}
    for left in method_names:
        for right in method_names:
            left_set = by_method.get(left, set())
            right_set = by_method.get(right, set())
            union = left_set | right_set
            matrix[left][right] = float(len(left_set & right_set) / len(union)) if union else float("nan")
    return matrix


def _render_and_save_report_figures(*, run_dir, summary, rows):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "accuracy_curves": seg._save_figure(
            _plot_mean_metric_curves(summary, metric_key="accuracy", title="Feature selection accuracy"),
            figure_dir / "accuracy_curves.png",
        ),
        "macro_f1_curves": seg._save_figure(
            _plot_mean_metric_curves(summary, metric_key="macro_f1", title="Feature selection macro-F1"),
            figure_dir / "macro_f1_curves.png",
        ),
        "accuracy_heatmaps": seg._save_figure(
            _plot_task_heatmaps(summary, rows, metric_key="accuracy", title_prefix="Accuracy"),
            figure_dir / "accuracy_heatmaps.png",
        ),
        "macro_f1_heatmaps": seg._save_figure(
            _plot_task_heatmaps(summary, rows, metric_key="macro_f1", title_prefix="Macro-F1"),
            figure_dir / "macro_f1_heatmaps.png",
        ),
        "selected_overlap": seg._save_figure(
            _plot_selected_overlap(summary),
            figure_dir / "selected_overlap.png",
        ),
    }
    return figures


def _plot_mean_metric_curves(summary, *, metric_key, title):
    metric_mean_key = f"{metric_key}_mean"
    metric_std_key = f"{metric_key}_std"
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    palette = plt.get_cmap("tab10")
    for method_idx, method_name in enumerate(summary["method_names"]):
        subset = [row for row in summary["summary_rows"] if row["method_name"] == method_name]
        subset.sort(key=lambda row: int(row["k"]))
        x = np.asarray([row["k"] for row in subset], dtype=np.int64)
        y = np.asarray([row[metric_mean_key] for row in subset], dtype=np.float64)
        std = np.asarray([row[metric_std_key] for row in subset], dtype=np.float64)
        color = palette(method_idx % 10)
        ax.plot(x, y, marker="o", linewidth=2, label=method_name, color=color)
        ax.fill_between(x, y - std, y + std, alpha=0.18, color=color)
    ax.set_xlabel("Selected filters (k)")
    ax.set_ylabel(metric_key.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return fig


def _plot_task_heatmaps(summary, rows, *, metric_key, title_prefix):
    method_names = summary["method_names"]
    task_names = [task["name"] for task in summary["tasks"]]
    k_values = summary["k_values"]
    fig, axes = plt.subplots(1, len(method_names), figsize=(4.4 * len(method_names), 2.6 + 0.55 * len(task_names)), constrained_layout=True)
    if len(method_names) == 1:
        axes = [axes]
    for ax, method_name in zip(axes, method_names):
        matrix = np.full((len(task_names), len(k_values)), np.nan, dtype=np.float64)
        for row_idx, task_name in enumerate(task_names):
            for col_idx, k in enumerate(k_values):
                matches = [row for row in rows if row["method_name"] == method_name and row["task_name"] == task_name and int(row["k"]) == int(k)]
                if matches:
                    matrix[row_idx, col_idx] = float(matches[0][metric_key])
        im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(k_values)))
        ax.set_xticklabels([str(k) for k in k_values])
        ax.set_yticks(np.arange(len(task_names)))
        ax.set_yticklabels(task_names if ax is axes[0] else [""] * len(task_names))
        ax.set_title(method_name)
        ax.set_xlabel("k")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                ax.text(col_idx, row_idx, "n/a" if value != value else f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.suptitle(f"{title_prefix} per task and k", fontsize=14)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    return fig


def _plot_selected_overlap(summary):
    task_names = [task["name"] for task in summary["tasks"]]
    method_names = summary["method_names"]
    n_cols = len(task_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 3.8), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]
    for ax, task_name in zip(axes, task_names):
        matrix = np.full((len(method_names), len(method_names)), np.nan, dtype=np.float64)
        overlap = summary["selected_overlap"][task_name]
        for row_idx, left in enumerate(method_names):
            for col_idx, right in enumerate(method_names):
                matrix[row_idx, col_idx] = float(overlap[left][right])
        im = ax.imshow(matrix, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_xticks(np.arange(len(method_names)))
        ax.set_xticklabels(method_names, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(method_names)))
        ax.set_yticklabels(method_names if ax is axes[0] else [""] * len(method_names))
        ax.set_title(task_name)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                ax.text(col_idx, row_idx, "n/a" if value != value else f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.suptitle(f"Selected-filter Jaccard overlap @k={summary['reference_k_for_overlap']}", fontsize=14)
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.03)
    return fig


def _build_report_markdown(summary, *, figures):
    lines = [
        "# Feature Selection Benchmark",
        "",
        "Classifier-only feature-selection benchmark inspired by Sec. 5.2 of *How Important Is a Neuron?*.",
        "",
        "## Configuration",
        "",
        f"- layer_name=`{summary['layer_name']}`",
        f"- n_steps=`{summary['n_steps']}`",
        f"- train_per_class=`{summary['train_per_class']}`",
        f"- eval_per_class=`{summary['eval_per_class']}`",
        f"- k_values=`{summary['k_values']}`",
        f"- channel_aggregation=`{summary['channel_aggregation']}`",
        f"- selection_rule=`{summary['selection_rule']}`",
        f"- random_seed=`{summary['random_seed']}`",
        "",
        "## Tasks",
        "",
        _build_task_table(summary["tasks"]),
        "",
        "## Best-k Summary",
        "",
        _build_best_table(summary["best_rows"]),
        "",
        "## Mean Accuracy / Macro-F1 by k",
        "",
        _build_summary_table(summary["summary_rows"]),
        "",
    ]
    for key in ("accuracy_curves", "macro_f1_curves", "accuracy_heatmaps", "macro_f1_heatmaps", "selected_overlap"):
        path = figures.get(key)
        if path:
            lines.extend([f"![]({seg._relative_markdown_path(path)})", ""])
    return "\n".join(lines)


def _build_task_table(tasks):
    lines = [
        "| Task | Classes |",
        "| --- | --- |",
    ]
    for task in tasks:
        lines.append(f"| {task['name']} | {', '.join(task['classes'])} |")
    return "\n".join(lines)


def _build_best_table(rows):
    lines = [
        "| Method | Best k | Mean accuracy | Mean macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_name']} | {row['best_k']} | {seg._format_number(row['best_accuracy_mean'])} | {seg._format_number(row['best_macro_f1_mean'])} |"
        )
    return "\n".join(lines)


def _build_summary_table(rows):
    lines = [
        "| Method | k | Mean accuracy | Std accuracy | Mean macro-F1 | Std macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method_name']} | {row['k']} | {seg._format_number(row['accuracy_mean'])} | {seg._format_number(row['accuracy_std'])} | {seg._format_number(row['macro_f1_mean'])} | {seg._format_number(row['macro_f1_std'])} |"
        )
    return "\n".join(lines)


def _default_method_name(spec):
    kind = str(spec.get("kind", "")).lower()
    if kind == "ig":
        return "IG"
    if kind == "naa":
        return "NAA"
    if kind == "cheap_ig":
        start = float(spec.get("segment_start", 0.0))
        end = float(spec.get("segment_end", 1.0))
        top_k = int(spec.get("selection_top_k", 0))
        fill_mode = spec.get("fill_mode", "zero")
        if fill_mode == "zero":
            suffix = "zero"
        else:
            suffix = f"{fill_mode}/rho{float(spec.get('fill_rho', 0.0)):g}"
        return f"Cheap-IG+[{start:g},{end:g}]/k{top_k}/{suffix}"
    return kind or "method"
