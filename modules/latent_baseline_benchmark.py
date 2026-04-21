from __future__ import annotations

"""ROAD-like latent baseline benchmark built on top of alpha-segment latent AOPC evaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from modules import alpha_segment_benchmark as seg
from modules.baseline_utils import DEFAULT_BLUR_SIGMA

DEFAULT_CACHE_ROOT = "output/latent_baseline_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_FILENAME = "latent_baseline_report.md"
DEFAULT_SUMMARY_JSON = "latent_baseline_summary.json"
DEFAULT_LAYER_NAME = "model.6"
DEFAULT_N_STEPS = 128
DEFAULT_BUDGET_PERCENTILES = tuple(range(1, 21))
DEFAULT_DONOR_KINDS = ("black_act",)
DEFAULT_PREVIEW_IMAGES = 5
DEFAULT_PREVIEW_GROUP_SIZE = 4


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
        variant = dict(variant)
        specs.append(classifier_method_spec("cheap_ig", **variant))
    return specs



def benchmark_classifier_latent_baseline(
    *,
    image_paths,
    method_specs,
    layer_name=DEFAULT_LAYER_NAME,
    n_steps=DEFAULT_N_STEPS,
    budget_percentiles=DEFAULT_BUDGET_PERCENTILES,
    donor_kinds=DEFAULT_DONOR_KINDS,
    blur_sigma=DEFAULT_BLUR_SIGMA,
    preview_images=DEFAULT_PREVIEW_IMAGES,
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
    normalized_method_specs = _normalize_method_specs(method_specs)
    donor_kinds = tuple(str(kind) for kind in donor_kinds)
    preview_images = int(max(0, min(int(preview_images), len(image_paths))))

    base_result = seg.benchmark_classifier_alpha_segment_latent_aopc(
        image_paths=image_paths,
        method_specs=normalized_method_specs,
        layer_name=layer_name,
        n_steps=n_steps,
        budget_percentiles=budget_percentiles,
        donor_kinds=donor_kinds,
        blur_sigma=blur_sigma,
        visual_images=0,
        output_dir=output_dir,
        cache_root=cache_root,
        save_output=False,
        target_dir=None,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        refresh_core=refresh_core,
        refresh_methods=refresh_methods,
        refresh_evaluations=refresh_evaluations,
        verbose=verbose,
    )

    summary = _build_summary(
        rows=base_result["rows"],
        method_rows=base_result["method_rows"],
        core_rows=base_result["core_rows"],
        method_specs=normalized_method_specs,
        donor_kinds=donor_kinds,
        budget_percentiles=base_result["budget_percentiles"],
        layer_name=layer_name,
        n_steps=n_steps,
        blur_sigma=blur_sigma,
        cache_root=cache_root,
        preview_image_paths=image_paths[:preview_images],
    )

    figures = {}
    preview_sections = []
    report_md = _build_report_markdown(summary, figures={}, preview_sections=[])
    report_path = None
    summary_path = None
    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = seg._prepare_existing_output_dir(target_dir)
        else:
            run_name = (
                f"latent_baseline_classifier_{seg._safe_slug(layer_name)}"
                f"_steps_{int(n_steps)}_images_{int(len(image_paths))}"
            )
            run_dir = seg._prepare_output_dir(output_dir, run_name)
        figures = _render_and_save_report_figures(run_dir=run_dir, summary=summary, rows=base_result["rows"])
        preview_sections = _render_preview_sections(
            run_dir=run_dir,
            summary=summary,
            method_rows=base_result["method_rows"],
        )
        report_md = _build_report_markdown(summary, figures=figures, preview_sections=preview_sections)
        report_path = run_dir / report_filename
        report_path.write_text(report_md + "\n", encoding="utf-8")
        summary_path = run_dir / summary_filename
        summary_path.write_text(
            seg._pretty_json(
                {
                    "summary": summary,
                    "rows": base_result["rows"],
                    "method_rows": base_result["method_rows"],
                    "core_rows": base_result["core_rows"],
                    "preview_sections": preview_sections,
                }
            ),
            encoding="utf-8",
        )

    return {
        "task": "classifier",
        "image_paths": image_paths,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "budget_percentiles": [int(v) for v in budget_percentiles],
        "donor_kinds": list(donor_kinds),
        "blur_sigma": float(blur_sigma),
        "method_specs": normalized_method_specs,
        "rows": base_result["rows"],
        "method_rows": base_result["method_rows"],
        "core_rows": base_result["core_rows"],
        "per_image": base_result["per_image"],
        "summary": summary,
        "report_markdown": report_md,
        "report_path": str(report_path) if report_path is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "figures": figures,
        "preview_sections": preview_sections,
        "output_dir": str(run_dir) if run_dir is not None else None,
        "cache_root": str(cache_root),
        "base_result": base_result,
    }



def render_latent_baseline_report(result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = result["summary"]
    figures = _render_and_save_report_figures(run_dir=output_dir, summary=summary, rows=result["rows"])
    preview_sections = _render_preview_sections(
        run_dir=output_dir,
        summary=summary,
        method_rows=result["method_rows"],
    )
    report_md = _build_report_markdown(summary, figures=figures, preview_sections=preview_sections)
    report_path = output_dir / DEFAULT_REPORT_FILENAME
    report_path.write_text(report_md + "\n", encoding="utf-8")
    summary_path = output_dir / DEFAULT_SUMMARY_JSON
    summary_path.write_text(
        seg._pretty_json(
            {
                "summary": summary,
                "rows": result["rows"],
                "method_rows": result["method_rows"],
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



def _normalize_method_specs(method_specs):
    normalized = []
    seen_names = set()
    for raw_spec in method_specs:
        spec = dict(raw_spec)
        name = str(spec.get("name") or _default_method_name(spec))
        if name in seen_names:
            raise ValueError(f"Duplicate method name: {name}")
        seen_names.add(name)
        kwargs = {
            k: v
            for k, v in spec.items()
            if k not in {"kind", "name", "family_name", "segment_start", "segment_end"}
        }
        normalized.append(
            seg.classifier_method_spec(
                spec["kind"],
                name=name,
                family_name=name,
                segment_start=float(spec.get("segment_start", 0.0)),
                segment_end=float(spec.get("segment_end", 1.0)),
                **kwargs,
            )
        )
    return normalized



def _build_summary(*, rows, method_rows, core_rows, method_specs, donor_kinds, budget_percentiles, layer_name, n_steps, blur_sigma, cache_root, preview_image_paths):
    method_names = [spec["name"] for spec in method_specs]
    by_method_donor = {}
    for row in rows:
        key = (row["method_name"], row["donor_kind"])
        by_method_donor.setdefault(key, []).append(row)

    summary_rows = []
    curve_summaries = {donor_kind: {} for donor_kind in donor_kinds}
    best_method_per_donor = {}
    best_donor_per_method = {}
    pairwise = {donor_kind: {} for donor_kind in donor_kinds}

    for donor_kind in donor_kinds:
        donor_rows = []
        for method_name in method_names:
            subset = by_method_donor.get((method_name, donor_kind), [])
            aoc_stats = seg._stats_record([row["aoc20"] for row in subset])
            aoc_norm_stats = seg._stats_record([row["aoc20_norm"] for row in subset])
            runtime_stats = seg._stats_record([row["method_duration_s"] for row in subset])
            eval_runtime_stats = seg._stats_record([row["evaluation_duration_s"] for row in subset])
            bench_runtime_stats = seg._stats_record([row["benchmark_duration_s"] for row in subset])
            abs_error_stats = seg._stats_record([row.get("abs_error") for row in subset])
            selected_stats = seg._stats_record([row.get("selected_neurons") for row in subset])
            drop_curves = np.asarray([row["drop_curve"] for row in subset], dtype=np.float64) if subset else np.empty((0, len(budget_percentiles)), dtype=np.float64)
            norm_curves = np.asarray([row["drop_curve_normalized"] for row in subset], dtype=np.float64) if subset else np.empty((0, len(budget_percentiles)), dtype=np.float64)
            curve_summaries[donor_kind][method_name] = {
                "budget_percentiles": [int(v) for v in budget_percentiles],
                "drop_curve_mean": [float(v) for v in np.nanmean(drop_curves, axis=0)] if drop_curves.size else [float("nan")] * len(budget_percentiles),
                "drop_curve_std": [float(v) for v in np.nanstd(drop_curves, axis=0)] if drop_curves.size else [float("nan")] * len(budget_percentiles),
                "drop_curve_norm_mean": [float(v) for v in np.nanmean(norm_curves, axis=0)] if norm_curves.size else [float("nan")] * len(budget_percentiles),
                "drop_curve_norm_std": [float(v) for v in np.nanstd(norm_curves, axis=0)] if norm_curves.size else [float("nan")] * len(budget_percentiles),
            }
            summary_row = {
                "method_name": method_name,
                "donor_kind": donor_kind,
                "aoc20_mean": float(aoc_stats["mean"]),
                "aoc20_std": float(aoc_stats["std"]),
                "aoc20_norm_mean": float(aoc_norm_stats["mean"]),
                "aoc20_norm_std": float(aoc_norm_stats["std"]),
                "runtime_s_mean": float(runtime_stats["mean"]),
                "runtime_s_std": float(runtime_stats["std"]),
                "evaluation_runtime_s_mean": float(eval_runtime_stats["mean"]),
                "benchmark_runtime_s_mean": float(bench_runtime_stats["mean"]),
                "abs_error_mean": float(abs_error_stats["mean"]),
                "selected_neurons_mean": float(selected_stats["mean"]),
                "n_images": int(len(subset)),
            }
            summary_rows.append(summary_row)
            donor_rows.append(summary_row)

        ordered = sorted(
            donor_rows,
            key=lambda row: (
                -float(row["aoc20_mean"]) if row["aoc20_mean"] == row["aoc20_mean"] else float("inf"),
                method_names.index(row["method_name"]),
            ),
        )
        best_method_per_donor[donor_kind] = ordered[0]["method_name"] if ordered else None

        for left_name in method_names:
            pairwise[donor_kind][left_name] = {}
            left_subset = {row["image_name"]: row for row in by_method_donor.get((left_name, donor_kind), [])}
            for right_name in method_names:
                right_subset = {row["image_name"]: row for row in by_method_donor.get((right_name, donor_kind), [])}
                common = sorted(set(left_subset) & set(right_subset))
                if not common:
                    pairwise[donor_kind][left_name][right_name] = float("nan")
                    continue
                wins = []
                for image_name in common:
                    left_score = float(left_subset[image_name]["aoc20"])
                    right_score = float(right_subset[image_name]["aoc20"])
                    wins.append(1.0 if left_score > right_score else 0.0)
                pairwise[donor_kind][left_name][right_name] = float(np.mean(np.asarray(wins, dtype=np.float64)))

    for method_name in method_names:
        donor_rows = [row for row in summary_rows if row["method_name"] == method_name]
        ordered = sorted(
            donor_rows,
            key=lambda row: (
                -float(row["aoc20_mean"]) if row["aoc20_mean"] == row["aoc20_mean"] else float("inf"),
                list(donor_kinds).index(row["donor_kind"]),
            ),
        )
        best_donor_per_method[method_name] = ordered[0]["donor_kind"] if ordered else None

    return {
        "task": "classifier",
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "budget_percentiles": [int(v) for v in budget_percentiles],
        "donor_kinds": list(donor_kinds),
        "blur_sigma": float(blur_sigma),
        "n_images": int(len(core_rows)),
        "cache_root": str(cache_root),
        "method_names": method_names,
        "summary_rows": summary_rows,
        "curve_summaries": curve_summaries,
        "best_method_per_donor": best_method_per_donor,
        "best_donor_per_method": best_donor_per_method,
        "pairwise_win_rates": pairwise,
        "core_summary": {
            "n_units_total": seg._stats_record([row["core"]["n_units_total"] for row in core_rows]),
            "core_runtime_s": seg._stats_record([row["core_duration_s"] for row in core_rows]),
        },
        "preview_image_paths": [str(path) for path in preview_image_paths],
    }



def _render_and_save_report_figures(*, run_dir, summary, rows):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    figures["heatmap_aoc20"] = seg._save_figure(_plot_method_donor_heatmap(summary, key="aoc20_mean", title="Mean latent AOC20", cbar_label="AOC20"), figure_dir / "latent_baseline_heatmap_aoc20.png")
    figures["heatmap_aoc20_norm"] = seg._save_figure(_plot_method_donor_heatmap(summary, key="aoc20_norm_mean", title="Mean latent AOC20 normalized", cbar_label="AOC20 / |clean logit|"), figure_dir / "latent_baseline_heatmap_aoc20_norm.png")
    for donor_kind in summary["donor_kinds"]:
        slug = seg._safe_slug(donor_kind)
        figures[f"summary_{slug}"] = seg._save_figure(_plot_summary_for_donor(summary, donor_kind), figure_dir / f"summary_{slug}.png")
        figures[f"dist_{slug}"] = seg._save_figure(_plot_distribution_for_donor(summary, rows, donor_kind), figure_dir / f"distribution_{slug}.png")
        figures[f"curve_{slug}"] = seg._save_figure(_plot_curves_for_donor(summary, donor_kind), figure_dir / f"curves_{slug}.png")
        figures[f"pairwise_{slug}"] = seg._save_figure(_plot_pairwise_for_donor(summary, donor_kind), figure_dir / f"pairwise_{slug}.png")
    return figures



def _render_preview_sections(*, run_dir, summary, method_rows):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    method_index = {(row["image_path"], row["method_name"]): row for row in method_rows}
    preview_methods = summary["method_names"]
    sections = []
    for page_idx, method_chunk in enumerate(seg._chunked(preview_methods, DEFAULT_PREVIEW_GROUP_SIZE), start=1):
        fig = _build_preview_figure(
            image_paths=summary["preview_image_paths"],
            method_names=method_chunk,
            method_index=method_index,
        )
        key = f"preview_page_{page_idx}"
        figure_path = figure_dir / f"{key}.png"
        saved_path = seg._save_figure(fig, figure_path)
        sections.append(
            {
                "page_idx": int(page_idx),
                "method_names": list(method_chunk),
                "figure_path": saved_path,
            }
        )
    return sections



def _plot_method_donor_heatmap(summary, *, key, title, cbar_label):
    method_names = summary["method_names"]
    donor_kinds = summary["donor_kinds"]
    matrix = np.full((len(method_names), len(donor_kinds)), np.nan, dtype=np.float64)
    for row in summary["summary_rows"]:
        row_idx = method_names.index(row["method_name"])
        col_idx = donor_kinds.index(row["donor_kind"])
        matrix[row_idx, col_idx] = float(row[key])
    fig, ax = plt.subplots(figsize=(2.0 + 1.4 * len(donor_kinds), 2.0 + 0.42 * len(method_names)), constrained_layout=True)
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(donor_kinds)))
    ax.set_xticklabels([seg._donor_label(kind) for kind in donor_kinds], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_yticklabels(method_names)
    ax.set_title(title)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            ax.text(c, r, "n/a" if value != value else f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    return fig



def _plot_summary_for_donor(summary, donor_kind):
    rows = [row for row in summary["summary_rows"] if row["donor_kind"] == donor_kind]
    rows = sorted(rows, key=lambda row: (-float(row["aoc20_mean"]) if row["aoc20_mean"] == row["aoc20_mean"] else float("inf"), row["method_name"]))
    labels = [row["method_name"] for row in rows]
    values = np.asarray([row["aoc20_mean"] for row in rows], dtype=np.float64)
    stds = np.asarray([row["aoc20_std"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.48 * len(rows) + 1.5)), constrained_layout=True)
    y = np.arange(len(rows))
    ax.errorbar(values, y, xerr=stds, fmt="o", color=seg._donor_color(donor_kind), ecolor="black", elinewidth=1.2, capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("AOC20")
    ax.set_title(f"Latent baseline summary: {seg._donor_label(donor_kind)}")
    ax.grid(axis="x", alpha=0.3)
    return fig



def _plot_distribution_for_donor(summary, rows, donor_kind):
    method_names = summary["method_names"]
    data = []
    labels = []
    for method_name in method_names:
        subset = [float(row["aoc20"]) for row in rows if row["method_name"] == method_name and row["donor_kind"] == donor_kind]
        if not subset:
            continue
        data.append(subset)
        labels.append(method_name)
    fig, ax = plt.subplots(figsize=(10.5, max(4.0, 0.48 * len(labels) + 1.5)), constrained_layout=True)
    ax.boxplot(data, vert=False, labels=labels, patch_artist=True, boxprops={"facecolor": seg._donor_color(donor_kind), "alpha": 0.55})
    ax.set_xlabel("Per-image AOC20")
    ax.set_title(f"Latent baseline distributions: {seg._donor_label(donor_kind)}")
    ax.grid(axis="x", alpha=0.3)
    return fig



def _plot_curves_for_donor(summary, donor_kind):
    x = np.asarray(summary["budget_percentiles"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    for method_name in summary["method_names"]:
        record = summary["curve_summaries"][donor_kind][method_name]
        y = np.asarray(record["drop_curve_mean"], dtype=np.float64)
        std = np.asarray(record["drop_curve_std"], dtype=np.float64)
        color = _method_color(method_name, summary["method_names"])
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=method_name)
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.18)
    ax.set_xlabel("Removed neurons (%)")
    ax.set_ylabel("Target logit drop")
    ax.set_title(f"Latent deletion curves: {seg._donor_label(donor_kind)}")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig



def _plot_pairwise_for_donor(summary, donor_kind):
    method_names = summary["method_names"]
    matrix = np.full((len(method_names), len(method_names)), np.nan, dtype=np.float64)
    pairwise = summary["pairwise_win_rates"][donor_kind]
    for row_idx, left_name in enumerate(method_names):
        for col_idx, right_name in enumerate(method_names):
            matrix[row_idx, col_idx] = float(pairwise[left_name][right_name])
    fig, ax = plt.subplots(figsize=(2.2 + 0.7 * len(method_names), 2.2 + 0.55 * len(method_names)), constrained_layout=True)
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(method_names)))
    ax.set_xticklabels(method_names, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(method_names)))
    ax.set_yticklabels(method_names)
    ax.set_title(f"Win-rate matrix: {seg._donor_label(donor_kind)}")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            ax.text(c, r, "n/a" if value != value else f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(row > col)")
    return fig



def _build_preview_figure(*, image_paths, method_names, method_index):
    n_rows = len(image_paths)
    n_cols = 1 + len(method_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_rows + 0.6),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle("Latent baseline benchmark previews", fontsize=14)
    for row_idx, image_path in enumerate(image_paths):
        _, image_np = seg.IG.load_image(str(image_path))
        raw_ax = axes[row_idx, 0]
        raw_ax.imshow(image_np, interpolation="nearest")
        raw_ax.axis("off")
        if row_idx == 0:
            raw_ax.set_title("Raw", fontsize=9)
        raw_ax.text(-0.02, 0.5, Path(image_path).name, transform=raw_ax.transAxes, va="center", ha="right", fontsize=8, rotation=90)
        for col_idx, method_name in enumerate(method_names, start=1):
            ax = axes[row_idx, col_idx]
            record = method_index.get((str(image_path), method_name))
            if record is None:
                ax.set_axis_off()
                continue
            overlay = seg._load_array_sidecar(record["overlay_map_path"])
            overlay = seg._normalize_map(overlay)
            if tuple(overlay.shape) != tuple(image_np.shape[:2]):
                overlay = seg.IG._resize_map_nearest(overlay, image_np.shape[:2]).astype(np.float32, copy=False)
            ax.imshow(image_np, interpolation="nearest")
            ax.imshow(overlay, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(method_name, fontsize=9)
    return fig



def _build_report_markdown(summary, *, figures, preview_sections):
    lines = [
        "# Latent Baseline Benchmark",
        "",
        "Classifier-only benchmark for `yolo11s-cls` using latent deletion AOC20 with baseline donors instead of ROAD imputation.",
        "",
        "## Configuration",
        "",
        f"- layer_name=`{summary['layer_name']}`",
        f"- n_steps=`{summary['n_steps']}`",
        f"- budget_percentiles=`{summary['budget_percentiles']}`",
        f"- donor_kinds=`{summary['donor_kinds']}`",
        f"- blur_sigma=`{summary['blur_sigma']}`",
        f"- n_images=`{summary['n_images']}`",
        "",
        "## Aggregate Heatmaps",
        "",
    ]
    for key in ("heatmap_aoc20", "heatmap_aoc20_norm"):
        path = figures.get(key)
        if path:
            lines.extend([f"![]({seg._relative_markdown_path(path)})", ""])

    lines.extend(["## Summary Table", "", _build_summary_table(summary["summary_rows"]), ""])

    for donor_kind in summary["donor_kinds"]:
        lines.extend([
            f"## {seg._donor_label(donor_kind)}",
            "",
            f"- best method by mean AOC20: `{summary['best_method_per_donor'].get(donor_kind)}`",
            "",
        ])
        for key_prefix in ("summary", "dist", "curve", "pairwise"):
            path = figures.get(f"{key_prefix}_{seg._safe_slug(donor_kind)}")
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



def _build_summary_table(rows):
    ordered = sorted(rows, key=lambda row: (list_key(row['donor_kind']), -row['aoc20_mean'] if row['aoc20_mean'] == row['aoc20_mean'] else float('inf'), row['method_name']))
    lines = [
        "| Donor | Method | AOC20 mean | AOC20 std | AOC20 norm mean | runtime_s mean | benchmark_runtime_s mean | abs_error mean |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {donor} | {method} | {aoc} | {aoc_std} | {aoc_norm} | {runtime} | {bench_runtime} | {abs_error} |".format(
                donor=seg._donor_label(row['donor_kind']),
                method=row['method_name'],
                aoc=seg._format_number(row['aoc20_mean']),
                aoc_std=seg._format_number(row['aoc20_std']),
                aoc_norm=seg._format_number(row['aoc20_norm_mean']),
                runtime=seg._format_number(row['runtime_s_mean']),
                bench_runtime=seg._format_number(row['benchmark_runtime_s_mean']),
                abs_error=seg._format_number(row['abs_error_mean']),
            )
        )
    return "\n".join(lines)



def _method_color(method_name, method_names):
    cmap = plt.get_cmap('tab20')
    idx = method_names.index(method_name) % max(1, cmap.N)
    return cmap(idx)



def _default_method_name(spec):
    kind = str(spec.get('kind', 'method'))
    if kind == 'ig':
        return 'IG'
    if kind == 'naa':
        return 'NAA'
    top_k = spec.get('selection_top_k')
    fill_mode = spec.get('fill_mode', 'zero')
    if fill_mode == 'naa_scaled':
        fill_suffix = f"naa_scaled/rho{float(spec.get('fill_rho', 0.8)):g}"
    else:
        fill_suffix = str(fill_mode)
    segment_start = float(spec.get('segment_start', 0.0))
    segment_end = float(spec.get('segment_end', 1.0))
    return f"Cheap-IG+[{segment_start:g},{segment_end:g}]/k{int(top_k)}/{fill_suffix}"



def list_key(donor_kind):
    order = list(seg._donor_order())
    return order.index(donor_kind) if donor_kind in order else len(order)
