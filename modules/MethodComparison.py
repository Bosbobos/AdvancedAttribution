import hashlib
from pathlib import Path
import re

import matplotlib.pyplot as plt

from modules import IG, IG_det, NAA, NAA_det


def _sanitize_name(value):
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    text = text.strip("_").lower()
    return text or "item"


def _normalize_image_paths(image_paths):
    return [Path(path) for path in image_paths]


def _short_id(*parts, length=10):
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def _build_image_filename(index, image_path, method_name, *id_parts):
    stem = _sanitize_name(image_path.stem)
    unique_id = _short_id(image_path, method_name, *id_parts)
    return f"{index:02d}_{stem}_{unique_id}_{method_name}.png"


def _build_markdown_table(rows):
    lines = [
        "| IG | NAA |",
        "| --- | --- |",
    ]
    for row in rows:
        lines.append(f"| ![[{row['ig_filename']}]] | ![[{row['naa_filename']}]] |")
    return "\n".join(lines)


def _relative_markdown_path(path):
    path = Path(path)
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _build_preview_markdown(rows):
    lines = [
        "| IG | NAA |",
        "| --- | --- |",
    ]
    for row in rows:
        ig_output_path = row.get("ig_output_path")
        naa_output_path = row.get("naa_output_path")
        if not ig_output_path or not naa_output_path:
            continue
        ig_ref = _relative_markdown_path(ig_output_path)
        naa_ref = _relative_markdown_path(naa_output_path)
        lines.append(f"| ![]({ig_ref}) | ![]({naa_ref}) |")
    return "\n".join(lines)


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


def _close_figure(fig):
    plt.close(fig)


def _render_comparison_figure(builder, image_np, cond_tensor, title):
    fig = builder(image_np, cond_tensor, title=title)
    if fig is None:
        raise RuntimeError("Не удалось построить итоговый spatial plot для сравнения методов.")
    return fig


def _run_classifier_methods(image_path, layer_name, n_steps, top_n, fd_eps, clear_every, verbose):
    ig_result = IG.run_conductance_pipeline(
        image_path=str(image_path),
        layer_name=layer_name,
        n_steps=n_steps,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
        verbose=verbose,
        show_total_plot=False,
        show_filter_plots=False,
    )
    naa_result = NAA.run_attribution_pipeline(
        image_path=str(image_path),
        layer_name=layer_name,
        n_steps=n_steps,
        top_n=top_n,
        clear_every=clear_every,
        verbose=verbose,
        show_total_plot=False,
        show_filter_plots=False,
    )
    return ig_result, naa_result


def _run_detector_methods(
    image_path,
    mode,
    layer_name,
    n_steps,
    top_n,
    roi_top_k,
    query_rank,
    query_head,
    bbox_iou_threshold,
    fd_eps,
    clear_every,
    verbose,
):
    ig_result = IG_det.run_detector_conductance(
        image_path=str(image_path),
        mode=mode,
        layer_name=layer_name,
        n_steps=n_steps,
        top_n=top_n,
        roi_top_k=roi_top_k,
        query_rank=query_rank,
        query_head=query_head,
        bbox_iou_threshold=bbox_iou_threshold,
        fd_eps=fd_eps,
        clear_every=clear_every,
        verbose=verbose,
        show_total_plot=False,
        show_filter_plots=False,
        show_target_box=False,
    )
    naa_result = NAA_det.run_attribution_pipeline(
        image_path=str(image_path),
        mode=mode,
        layer_name=layer_name,
        n_steps=n_steps,
        top_n=top_n,
        roi_top_k=roi_top_k,
        query_rank=query_rank,
        query_head=query_head,
        bbox_iou_threshold=bbox_iou_threshold,
        clear_every=clear_every,
        verbose=verbose,
        show_total_plot=False,
        show_filter_plots=False,
        show_target_box=False,
    )
    return ig_result, naa_result


def compare_classifiers(
    image_paths,
    layer_name,
    n_steps,
    save_output=False,
    output_dir="output",
    top_n=0,
    fd_eps=1e-3,
    clear_every=8,
    verbose=False,
    target_dir=None,
    markdown_filename="comparison.md",
):
    image_paths = _normalize_image_paths(image_paths)
    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = _prepare_existing_output_dir(target_dir)
        else:
            run_name = f"classifiers_{_sanitize_name(layer_name)}_steps_{n_steps}"
            run_dir = _prepare_output_dir(output_dir, run_name)

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        ig_result, naa_result = _run_classifier_methods(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            verbose=verbose,
        )

        file_id_parts = (
            "classifiers",
            layer_name,
            n_steps,
            top_n,
            fd_eps,
            clear_every,
        )
        ig_filename = _build_image_filename(index, image_path, "ig", *file_id_parts)
        naa_filename = _build_image_filename(index, image_path, "naa", *file_id_parts)

        ig_fig = _render_comparison_figure(
            IG.build_total_conductance_overlay_figure,
            ig_result["image_np"],
            ig_result["cond_tensor"],
            ig_result["total_plot_title"],
        )
        naa_fig = _render_comparison_figure(
            NAA.build_total_attribution_overlay_figure,
            naa_result["image_np"],
            naa_result["cond_tensor"],
            naa_result["total_plot_title"],
        )

        ig_output_path = None
        naa_output_path = None
        if run_dir is not None:
            ig_output_path = run_dir / ig_filename
            naa_output_path = run_dir / naa_filename
            _save_figure(ig_fig, ig_output_path)
            _save_figure(naa_fig, naa_output_path)
        else:
            _close_figure(ig_fig)
            _close_figure(naa_fig)

        rows.append(
            {
                "image_path": str(image_path),
                "ig_filename": ig_filename,
                "naa_filename": naa_filename,
                "ig_output_path": str(ig_output_path) if ig_output_path is not None else None,
                "naa_output_path": str(naa_output_path) if naa_output_path is not None else None,
                "ig_result": ig_result,
                "naa_result": naa_result,
            }
        )

    markdown_table = _build_markdown_table(rows)
    preview_markdown = _build_preview_markdown(rows)
    markdown_path = None
    if run_dir is not None and markdown_filename:
        markdown_path = run_dir / markdown_filename
        markdown_path.write_text(markdown_table + "\n", encoding="utf-8")

    return {
        "rows": rows,
        "markdown_table": markdown_table,
        "preview_markdown": preview_markdown,
        "output_dir": str(run_dir) if run_dir is not None else None,
        "markdown_path": str(markdown_path) if markdown_path is not None else None,
    }


def compare_detectors(
    image_paths,
    mode,
    layer_name,
    n_steps,
    roi_top_k,
    query_rank=None,
    query_head=None,
    bbox_iou_threshold=IG_det.BBOX_RANK_IOU_THRESHOLD,
    save_output=False,
    output_dir="output",
    top_n=0,
    fd_eps=1e-3,
    clear_every=8,
    verbose=False,
    target_dir=None,
    markdown_filename="comparison.md",
):
    supported_modes = {"fixed_roi_lse", "fixed_roi_logmeanexp", "fixed_roi_mean"}
    if mode not in supported_modes:
        raise ValueError(f"compare_detectors поддерживает только mode из {tuple(sorted(supported_modes))}, получено {mode!r}")

    image_paths = _normalize_image_paths(image_paths)
    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = _prepare_existing_output_dir(target_dir)
        else:
            run_name = f"detectors_{_sanitize_name(mode)}_{_sanitize_name(layer_name)}_steps_{n_steps}"
            run_dir = _prepare_output_dir(output_dir, run_name)

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        ig_result, naa_result = _run_detector_methods(
            image_path=image_path,
            mode=mode,
            layer_name=layer_name,
            n_steps=n_steps,
            top_n=top_n,
            roi_top_k=roi_top_k,
            query_rank=query_rank,
            query_head=query_head,
            bbox_iou_threshold=bbox_iou_threshold,
            fd_eps=fd_eps,
            clear_every=clear_every,
            verbose=verbose,
        )

        file_id_parts = (
            "detectors",
            mode,
            layer_name,
            n_steps,
            roi_top_k,
            query_rank,
            query_head,
            bbox_iou_threshold,
            top_n,
            fd_eps,
            clear_every,
        )
        ig_filename = _build_image_filename(index, image_path, "ig", *file_id_parts)
        naa_filename = _build_image_filename(index, image_path, "naa", *file_id_parts)

        ig_fig = _render_comparison_figure(
            IG_det.build_total_conductance_overlay_figure,
            ig_result["image_np"],
            ig_result["cond_tensor"],
            ig_result["total_plot_title"],
        )
        naa_fig = _render_comparison_figure(
            NAA_det.build_total_attribution_overlay_figure,
            naa_result["image_np"],
            naa_result["cond_tensor"],
            naa_result["total_plot_title"],
        )

        ig_output_path = None
        naa_output_path = None
        if run_dir is not None:
            ig_output_path = run_dir / ig_filename
            naa_output_path = run_dir / naa_filename
            _save_figure(ig_fig, ig_output_path)
            _save_figure(naa_fig, naa_output_path)
        else:
            _close_figure(ig_fig)
            _close_figure(naa_fig)

        rows.append(
            {
                "image_path": str(image_path),
                "ig_filename": ig_filename,
                "naa_filename": naa_filename,
                "ig_output_path": str(ig_output_path) if ig_output_path is not None else None,
                "naa_output_path": str(naa_output_path) if naa_output_path is not None else None,
                "ig_result": ig_result,
                "naa_result": naa_result,
            }
        )

    markdown_table = _build_markdown_table(rows)
    preview_markdown = _build_preview_markdown(rows)
    markdown_path = None
    if run_dir is not None and markdown_filename:
        markdown_path = run_dir / markdown_filename
        markdown_path.write_text(markdown_table + "\n", encoding="utf-8")

    return {
        "rows": rows,
        "markdown_table": markdown_table,
        "preview_markdown": preview_markdown,
        "output_dir": str(run_dir) if run_dir is not None else None,
        "markdown_path": str(markdown_path) if markdown_path is not None else None,
    }


def export_current_notebook_tables(
    image_paths,
    n_steps,
    output_dir="output",
    detector_mode="fixed_roi_mean",
    detector_layers=("model.0", "model.22"),
    classifier_layers=("model.0", "model.6"),
    detector_roi_top_k=3,
    top_n=0,
    fd_eps=1e-3,
    clear_every=8,
    verbose=False,
    markdown_filename="all_tables.md",
):
    image_paths = _normalize_image_paths(image_paths)
    run_name = f"all_tables_steps_{n_steps}"
    run_dir = _prepare_output_dir(output_dir, run_name)

    sections = []

    for layer_name in detector_layers:
        result = compare_detectors(
            image_paths=image_paths,
            mode=detector_mode,
            layer_name=layer_name,
            n_steps=n_steps,
            roi_top_k=detector_roi_top_k,
            save_output=True,
            output_dir=output_dir,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            verbose=verbose,
            target_dir=run_dir,
            markdown_filename=None,
        )
        title = f"Detectors, layer={layer_name}, mode={detector_mode}"
        sections.append(
            {
                "title": title,
                "kind": "detector",
                "result": result,
                "markdown_table": result["markdown_table"],
                "preview_markdown": result["preview_markdown"],
            }
        )

    for layer_name in classifier_layers:
        result = compare_classifiers(
            image_paths=image_paths,
            layer_name=layer_name,
            n_steps=n_steps,
            save_output=True,
            output_dir=output_dir,
            top_n=top_n,
            fd_eps=fd_eps,
            clear_every=clear_every,
            verbose=verbose,
            target_dir=run_dir,
            markdown_filename=None,
        )
        title = f"Classifiers, layer={layer_name}"
        sections.append(
            {
                "title": title,
                "kind": "classifier",
                "result": result,
                "markdown_table": result["markdown_table"],
                "preview_markdown": result["preview_markdown"],
            }
        )

    markdown_blocks = []
    preview_blocks = []
    for section in sections:
        markdown_blocks.append(f"## {section['title']}\n\n{section['markdown_table']}")
        preview_blocks.append(f"## {section['title']}\n\n{section['preview_markdown']}")

    combined_markdown = "\n\n".join(markdown_blocks) + "\n"
    combined_preview_markdown = "\n\n".join(preview_blocks)

    markdown_path = run_dir / markdown_filename
    markdown_path.write_text(combined_markdown, encoding="utf-8")

    return {
        "sections": sections,
        "markdown_table": combined_markdown,
        "preview_markdown": combined_preview_markdown,
        "output_dir": str(run_dir),
        "markdown_path": str(markdown_path),
    }
