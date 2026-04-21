from __future__ import annotations

"""ImageNet feature-selection benchmark aligned with How Important Is a Neuron? Sec. 5.2."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules import IG
from modules import alpha_segment_benchmark as seg
from modules import feature_selection_benchmark as fs


DEFAULT_CACHE_ROOT = "output/imagenet_feature_selection_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_FILENAME = "imagenet_feature_selection_report.md"
DEFAULT_SUMMARY_JSON = "imagenet_feature_selection_summary.json"
DEFAULT_LAYER_NAME = "model.6"
DEFAULT_N_STEPS = 128
DEFAULT_TRAIN_PER_CLASS = 30
DEFAULT_EVAL_PER_CLASS = 20
DEFAULT_K_VALUES = (4, 8, 16, 32, 64)
DEFAULT_RANDOM_SEED = 0
DEFAULT_CHANNEL_AGGREGATION = "positive_mean"
DEFAULT_SELECTION_RULE = "max_over_classes"
DEFAULT_IMAGENET_VAL_ROOT = "imagenet_val"


def classifier_method_spec(kind, name=None, **kwargs):
    return fs.classifier_method_spec(kind, name=name, **kwargs)


def default_classifier_method_specs():
    return fs.default_classifier_method_specs()


def collect_imagenet_val_image_paths(root=DEFAULT_IMAGENET_VAL_ROOT):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"ImageNet val root not found: {root}")
    image_paths = []
    for class_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name):
        for image_path in sorted([path for path in class_dir.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}], key=lambda path: path.name.lower()):
            image_paths.append(str(image_path))
    return image_paths


def default_imagenet_feature_selection_tasks(root=DEFAULT_IMAGENET_VAL_ROOT, *, random_seed=DEFAULT_RANDOM_SEED):
    wnids = _sorted_wnids(root)
    if len(wnids) != len(IG.class_names):
        raise ValueError(f"Expected 1000 wnids and 1000 classifier names, got {len(wnids)} and {len(IG.class_names)}")

    related_dogs_idx = [151, 152, 153, 154, 155]
    related_cats_idx = [281, 282, 283, 284, 285]
    reserved = set(related_dogs_idx + related_cats_idx)
    rng = np.random.default_rng(int(random_seed))
    remaining = np.asarray([idx for idx in range(len(wnids)) if idx not in reserved], dtype=np.int64)
    random_selection = rng.choice(remaining, size=10, replace=False)

    tasks = [
        _task_from_indices("dogs_related", related_dogs_idx, wnids),
        _task_from_indices("cats_related", related_cats_idx, wnids),
        _task_from_indices("random_a", random_selection[:5].tolist(), wnids),
        _task_from_indices("random_b", random_selection[5:].tolist(), wnids),
    ]
    return tasks


def benchmark_imagenet_feature_selection(
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

    by_label = _group_imagenet_images_by_label(image_paths)
    wnids = sorted(by_label)
    wnid_to_index = {wnid: idx for idx, wnid in enumerate(wnids)}
    if len(wnid_to_index) != len(IG.class_names):
        raise ValueError(
            f"Expected wnid count to match classifier class count. Got wnids={len(wnid_to_index)} class_names={len(IG.class_names)}"
        )

    if class_tasks is None:
        class_tasks = default_imagenet_feature_selection_tasks(random_seed=random_seed)
    normalized_tasks = _normalize_imagenet_tasks(
        class_tasks,
        by_label=by_label,
        wnid_to_index=wnid_to_index,
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
                f"imagenet_feature_selection_{seg._safe_slug(layer_name)}"
                f"_tasks{len(normalized_tasks)}_k{seg._safe_slug('-'.join(str(v) for v in k_values))}"
            )
            run_dir = seg._prepare_output_dir(output_dir, run_name)

    used_images = sorted({path for task in normalized_tasks for path in task["train_images"] + task["eval_images"]})
    core_rows = []
    method_rows = []
    core_by_image = {}
    method_by_image = {}
    for image_path in used_images:
        wnid = _imagenet_label_from_path(image_path)
        target_class_override = int(wnid_to_index[wnid])
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
                f"Failed to build ImageNet feature-selection core for {image_path}: {core_record.get('error')} "
                f"(cache={core_record.get('cache_path')})"
            )
        core_value = core_record["value"]
        gap_features = fs._gap_features_from_core(core_value)
        core_row = {
            "image_path": image_path,
            "image_name": Path(image_path).name,
            "wnid": wnid,
            "target_class_override": int(target_class_override),
            "target_name_override": IG.class_names[target_class_override],
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
            "target_class_override": int(target_class_override),
        }

        method_by_image[image_path] = {}
        for method_spec in normalized_method_specs:
            method_record = seg._load_method_record(
                image_path=image_path,
                layer_name=layer_name,
                n_steps=n_steps,
                method_spec=method_spec,
                target_class_override=target_class_override,
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
            channel_scores = fs._channel_scores_from_method(
                method_value=method_value,
                activation_shape=core_by_image[image_path]["activation_shape"],
            )
            method_row = {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "wnid": wnid,
                "target_class_override": int(target_class_override),
                "target_name_override": IG.class_names[target_class_override],
                "method_name": method_spec["name"],
                "method_id": method_spec["id"],
                "kind": method_spec["kind"],
                "method_cache_path": method_record["cache_path"],
                "method_from_cache": method_record["from_cache"],
                "method_duration_s": method_record["duration_s"],
                "unit_scores_path": method_value["unit_scores_path"],
                "abs_error": float(method_value.get("abs_error", float("nan"))),
                "selected_neurons": method_value.get("selected_neurons"),
                "target_class": int(method_value.get("target_class", -1)),
                "target_name": method_value.get("target_name"),
                "channel_scores": channel_scores.tolist(),
            }
            method_rows.append(method_row)
            method_by_image[image_path][method_spec["name"]] = {
                "row": method_row,
                "channel_scores": channel_scores,
            }

    rows = []
    for task in normalized_tasks:
        class_to_train = {wnid: list(paths) for wnid, paths in task["train_by_class"].items()}
        class_to_eval = {wnid: list(paths) for wnid, paths in task["eval_by_class"].items()}
        wnids = list(task["wnids"])
        label_to_index = {wnid: idx for idx, wnid in enumerate(wnids)}

        for method_spec in normalized_method_specs:
            per_class_scores = {}
            for wnid in wnids:
                vectors = [method_by_image[path][method_spec["name"]]["channel_scores"] for path in class_to_train[wnid]]
                per_class_scores[wnid] = fs._aggregate_class_channel_scores(
                    vectors,
                    aggregation_mode=channel_aggregation,
                )
            global_scores = fs._select_global_channel_scores(
                per_class_scores,
                selection_rule=selection_rule,
            )

            train_features = {path: core_by_image[path]["gap_features"] for wnid in wnids for path in class_to_train[wnid]}
            eval_features = {path: core_by_image[path]["gap_features"] for wnid in wnids for path in class_to_eval[wnid]}

            for k in k_values:
                selected_channels = fs._topk_indices(global_scores, k)
                x_train = np.stack([train_features[path][selected_channels] for wnid in wnids for path in class_to_train[wnid]], axis=0)
                y_train = np.asarray([label_to_index[wnid] for wnid in wnids for path in class_to_train[wnid]], dtype=np.int64)
                x_eval = np.stack([eval_features[path][selected_channels] for wnid in wnids for path in class_to_eval[wnid]], axis=0)
                y_eval = np.asarray([label_to_index[wnid] for wnid in wnids for path in class_to_eval[wnid]], dtype=np.int64)
                metrics = fs._train_and_evaluate_linear_probe(
                    x_train=x_train,
                    y_train=y_train,
                    x_eval=x_eval,
                    y_eval=y_eval,
                    random_seed=int(random_seed),
                )
                rows.append(
                    {
                        "task_name": task["name"],
                        "task_wnids": list(wnids),
                        "task_class_names": [task["wnid_to_name"][wnid] for wnid in wnids],
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
        figures = fs._render_and_save_report_figures(run_dir=run_dir, summary=summary, rows=rows)
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
        "dataset": "imagenet_val",
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


def render_imagenet_feature_selection_report(result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = fs._render_and_save_report_figures(run_dir=output_dir, summary=result["summary"], rows=result["rows"])
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


def _sorted_wnids(root):
    root = Path(root)
    return sorted([path.name for path in root.iterdir() if path.is_dir()])


def _imagenet_label_from_path(path):
    return Path(path).parent.name


def _normalize_imagenet_tasks(class_tasks, *, by_label, wnid_to_index, train_per_class, eval_per_class, random_seed):
    tasks = []
    for task_idx, task in enumerate(class_tasks):
        name = str(task["name"])
        wnids = [str(wnid) for wnid in task["wnids"]]
        if len(wnids) != 5:
            raise ValueError(f"Task {name} must contain exactly 5 wnids, got {wnids}")
        train_by_class = {}
        eval_by_class = {}
        wnid_to_name = {}
        for class_idx, wnid in enumerate(wnids):
            if wnid not in by_label:
                raise ValueError(f"Wnid {wnid} from task {name} not found in dataset.")
            paths = list(by_label[wnid])
            required = int(train_per_class) + int(eval_per_class)
            if len(paths) < required:
                raise ValueError(f"Task {name} requires {required} images for {wnid}, got {len(paths)}.")
            rng = np.random.default_rng(int(random_seed) + 1000 * task_idx + class_idx)
            order = rng.permutation(len(paths))
            ordered_paths = [paths[int(idx)] for idx in order.tolist()]
            train_by_class[wnid] = ordered_paths[: int(train_per_class)]
            eval_by_class[wnid] = ordered_paths[int(train_per_class) : int(train_per_class) + int(eval_per_class)]
            wnid_to_name[wnid] = IG.class_names[int(wnid_to_index[wnid])]
        tasks.append(
            {
                "name": name,
                "wnids": wnids,
                "wnid_to_name": wnid_to_name,
                "train_by_class": train_by_class,
                "eval_by_class": eval_by_class,
                "train_images": [path for wnid in wnids for path in train_by_class[wnid]],
                "eval_images": [path for wnid in wnids for path in eval_by_class[wnid]],
            }
        )
    return tasks


def _group_imagenet_images_by_label(image_paths):
    by_label = {}
    for image_path in image_paths:
        wnid = _imagenet_label_from_path(image_path)
        by_label.setdefault(wnid, []).append(str(image_path))
    for wnid in list(by_label):
        by_label[wnid] = sorted(by_label[wnid], key=lambda path: Path(path).name.lower())
    return by_label


def _task_from_indices(name, indices, wnids):
    return {"name": str(name), "wnids": [str(wnids[int(idx)]) for idx in indices]}


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
    summary = fs._build_summary(
        rows=rows,
        core_rows=core_rows,
        method_rows=method_rows,
        method_specs=method_specs,
        tasks=[{"name": task["name"], "classes": [task["wnid_to_name"][wnid] for wnid in task["wnids"]]} for task in tasks],
        layer_name=layer_name,
        n_steps=n_steps,
        k_values=k_values,
        train_per_class=train_per_class,
        eval_per_class=eval_per_class,
        channel_aggregation=channel_aggregation,
        selection_rule=selection_rule,
        random_seed=random_seed,
        cache_root=cache_root,
    )
    summary["dataset"] = "imagenet_val"
    summary["tasks"] = [
        {
            "name": task["name"],
            "wnids": list(task["wnids"]),
            "classes": [task["wnid_to_name"][wnid] for wnid in task["wnids"]],
        }
        for task in tasks
    ]
    return summary


def _build_report_markdown(summary, *, figures):
    lines = [
        "# ImageNet Feature Selection Benchmark",
        "",
        "Classifier-only feature-selection benchmark aligned with Sec. 5.2 of *How Important Is a Neuron?*.",
        "",
        "## Configuration",
        "",
        f"- dataset=`{summary.get('dataset', 'imagenet_val')}`",
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
        fs._build_best_table(summary["best_rows"]),
        "",
        "## Mean Accuracy / Macro-F1 by k",
        "",
        fs._build_summary_table(summary["summary_rows"]),
        "",
    ]
    for key in ("accuracy_curves", "macro_f1_curves", "accuracy_heatmaps", "macro_f1_heatmaps", "selected_overlap"):
        path = figures.get(key)
        if path:
            lines.extend([f"![]({seg._relative_markdown_path(path)})", ""])
    return "\n".join(lines)


def _build_task_table(tasks):
    lines = [
        "| Task | WNIDs | Classes |",
        "| --- | --- | --- |",
    ]
    for task in tasks:
        lines.append(
            f"| {task['name']} | {', '.join(task['wnids'])} | {', '.join(task['classes'])} |"
        )
    return "\n".join(lines)
