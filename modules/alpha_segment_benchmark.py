from __future__ import annotations

"""Alpha-segment latent AOPC benchmark for classifier attribution methods."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from modules import IG, NAA, cheap_ig
from modules.baseline_utils import DEFAULT_BLUR_SIGMA, build_image_baseline
from modules.method_timing_cache import current_device_label, image_signature, load_or_compute_cached_value

try:
    import scipy.sparse as scipy_sparse
    from scipy.sparse.linalg import spsolve as scipy_spsolve

    SCIPY_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime environment dependent.
    scipy_sparse = None
    scipy_spsolve = None
    SCIPY_IMPORT_ERROR = exc


DEFAULT_CACHE_ROOT = "output/alpha_segment_cache"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_REPORT_FILENAME = "alpha_segment_report.md"
DEFAULT_SUMMARY_JSON = "alpha_segment_summary.json"
DEFAULT_LAYER_NAME = "model.6"
DEFAULT_N_STEPS = 192
DEFAULT_SEGMENT_END_VALUES = tuple(float(value) for value in np.linspace(0.1, 1.0, 10))
DEFAULT_BUDGET_PERCENTILES = tuple(range(1, 21))
DEFAULT_DONOR_KINDS = (
    "zero_baseline",
    "black_act",
    "blur_act",
    "layer_mean_exclusive",
    "spatial_nli_same_channel",
)
DEFAULT_BLUR_SIGMA = 16.0
DEFAULT_VISUAL_IMAGES = 10
DEFAULT_VISUAL_PAGE_SIZE = 5
DEFAULT_EPS = 1e-12
_ORTHOGONAL_WEIGHT = 1.0 / 6.0
_DIAGONAL_WEIGHT = 1.0 / 12.0
NLI_NEIGHBOR_WEIGHTS = (
    ((-1, 0), _ORTHOGONAL_WEIGHT),
    ((1, 0), _ORTHOGONAL_WEIGHT),
    ((0, -1), _ORTHOGONAL_WEIGHT),
    ((0, 1), _ORTHOGONAL_WEIGHT),
    ((-1, -1), _DIAGONAL_WEIGHT),
    ((-1, 1), _DIAGONAL_WEIGHT),
    ((1, -1), _DIAGONAL_WEIGHT),
    ((1, 1), _DIAGONAL_WEIGHT),
)


def classifier_method_spec(kind, name=None, family_name=None, **kwargs):
    spec = {"kind": str(kind)}
    spec.update(kwargs)
    spec["family_name"] = str(family_name) if family_name is not None else _default_family_name(spec["kind"])
    spec["name"] = str(name) if name is not None else _default_method_name(spec)
    return spec


def default_classifier_method_specs(*, segment_end_values=DEFAULT_SEGMENT_END_VALUES, cheap_ig_config=None):
    cheap_ig_config = dict(cheap_ig_config or {})
    cheap_kwargs = {
        "selection_mode": cheap_ig_config.get("selection_mode", "positive"),
        "selection_top_k": int(cheap_ig_config.get("selection_top_k", 4000)),
        "fill_mode": str(cheap_ig_config.get("fill_mode", "naa_scaled")),
        "fill_rho": float(cheap_ig_config.get("fill_rho", 0.6)),
    }
    specs = []
    for segment_end in segment_end_values:
        segment_end = float(segment_end)
        specs.append(
            classifier_method_spec(
                "ig",
                family_name="IG",
                segment_start=0.0,
                segment_end=segment_end,
            )
        )
        specs.append(
            classifier_method_spec(
                "naa",
                family_name="NAA",
                segment_start=0.0,
                segment_end=segment_end,
            )
        )
        specs.append(
            classifier_method_spec(
                "cheap_ig",
                family_name="Cheap-IG",
                segment_start=0.0,
                segment_end=segment_end,
                **cheap_kwargs,
            )
        )
    return specs


def benchmark_classifier_alpha_segment_latent_aopc(
    *,
    image_paths,
    method_specs,
    layer_name=DEFAULT_LAYER_NAME,
    n_steps=DEFAULT_N_STEPS,
    budget_percentiles=DEFAULT_BUDGET_PERCENTILES,
    donor_kinds=DEFAULT_DONOR_KINDS,
    blur_sigma=DEFAULT_BLUR_SIGMA,
    visual_images=DEFAULT_VISUAL_IMAGES,
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
    _require_scipy()

    image_paths = [str(Path(path)) for path in image_paths]
    donor_kinds = _normalize_donor_kinds(donor_kinds)
    budget_percentiles = _normalize_budget_percentiles(budget_percentiles)
    blur_sigma = float(blur_sigma)
    normalized_method_specs = _normalize_method_specs(method_specs)
    segment_end_values = sorted(
        {
            float(spec.get("segment_end", 1.0))
            for spec in normalized_method_specs
        }
    )
    visual_images = int(max(0, min(int(visual_images), len(image_paths))))
    visual_image_paths = image_paths[:visual_images]
    single_image_path = image_paths[0] if image_paths else None

    run_dir = None
    if save_output:
        if target_dir is not None:
            run_dir = _prepare_existing_output_dir(target_dir)
        else:
            run_name = (
                f"alpha_segment_latent_aopc_{_safe_slug(layer_name)}"
                f"_steps_{int(n_steps)}_images_{int(len(image_paths))}"
            )
            run_dir = _prepare_output_dir(output_dir, run_name)

    core_rows = []
    method_rows = []
    rows = []
    per_image = []

    for image_path in image_paths:
        core_record = _load_core_record(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            blur_sigma=blur_sigma,
            budget_percentiles=budget_percentiles,
            cache_root=cache_root,
            refresh=refresh_core,
        )
        if core_record.get("error") is not None or core_record.get("value") is None:
            raise RuntimeError(
                f"Failed to build alpha-segment core for {image_path}: {core_record.get('error')} "
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
                    "family_name": method_spec["family_name"],
                    "segment_end": float(method_spec.get("segment_end", 1.0)),
                    "method_cache_path": method_record["cache_path"],
                    "method_from_cache": method_record["from_cache"],
                    "method_duration_s": method_record["duration_s"],
                    **method_value,
                }
            )
            image_payload["methods"][method_spec["name"]] = {
                "cache_path": method_record["cache_path"],
                "from_cache": method_record["from_cache"],
                "duration_s": method_record["duration_s"],
                "method": method_value,
                "evaluations": {},
            }

            for donor_kind in donor_kinds:
                evaluation_record = _load_evaluation_record(
                    image_path=image_path,
                    layer_name=layer_name,
                    n_steps=n_steps,
                    method_spec=method_spec,
                    core_value=core_value,
                    method_value=method_value,
                    donor_kind=donor_kind,
                    budget_percentiles=budget_percentiles,
                    blur_sigma=blur_sigma,
                    cache_root=cache_root,
                    refresh=refresh_evaluations,
                )
                if evaluation_record.get("error") is not None or evaluation_record.get("value") is None:
                    raise RuntimeError(
                        f"Failed to evaluate {method_spec['name']} with donor={donor_kind} for {image_path}: "
                        f"{evaluation_record.get('error')} (cache={evaluation_record.get('cache_path')})"
                    )
                evaluation = evaluation_record["value"]
                row = {
                    "image_path": image_path,
                    "image_name": Path(image_path).name,
                    "method_name": method_spec["name"],
                    "method_id": method_spec["id"],
                    "family_name": method_spec["family_name"],
                    "kind": method_spec["kind"],
                    "segment_start": float(method_spec.get("segment_start", 0.0)),
                    "segment_end": float(method_spec.get("segment_end", 1.0)),
                    "donor_kind": donor_kind,
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
                image_payload["methods"][method_spec["name"]]["evaluations"][donor_kind] = {
                    "cache_path": evaluation_record["cache_path"],
                    "from_cache": evaluation_record["from_cache"],
                    "duration_s": evaluation_record["duration_s"],
                    "evaluation": evaluation,
                }

        per_image.append(image_payload)

    summary = _build_summary(
        rows=rows,
        core_rows=core_rows,
        method_rows=method_rows,
        method_specs=normalized_method_specs,
        layer_name=layer_name,
        n_steps=n_steps,
        budget_percentiles=budget_percentiles,
        donor_kinds=donor_kinds,
        segment_end_values=segment_end_values,
        blur_sigma=blur_sigma,
        cache_root=cache_root,
        single_image_path=single_image_path,
        visual_image_paths=visual_image_paths,
    )

    figures = {}
    visual_sections = []
    if run_dir is not None:
        figures = _render_and_save_report_figures(
            run_dir=run_dir,
            summary=summary,
            rows=rows,
            method_rows=method_rows,
            method_specs=normalized_method_specs,
        )
        visual_sections = _render_visual_sections(
            run_dir=run_dir,
            summary=summary,
            method_rows=method_rows,
            visual_image_paths=visual_image_paths,
            method_specs=normalized_method_specs,
        )
        report_md = _build_report_markdown(
            summary,
            figures=figures,
            visual_sections=visual_sections,
            method_specs=normalized_method_specs,
        )
        report_path = run_dir / report_filename
        report_path.write_text(report_md + "\n", encoding="utf-8")
        summary_path = run_dir / summary_filename
        summary_path.write_text(
            _pretty_json(
                {
                    "summary": summary,
                    "rows": rows,
                    "method_rows": method_rows,
                    "core_rows": core_rows,
                    "visual_sections": visual_sections,
                }
            ),
            encoding="utf-8",
        )
    else:
        report_md = _build_report_markdown(summary, figures={}, visual_sections=[], method_specs=normalized_method_specs)
        report_path = None
        summary_path = None

    return {
        "task": "classifier",
        "image_paths": image_paths,
        "layer_name": layer_name,
        "n_steps": int(n_steps),
        "budget_percentiles": [int(v) for v in budget_percentiles],
        "donor_kinds": list(donor_kinds),
        "blur_sigma": float(blur_sigma),
        "method_specs": normalized_method_specs,
        "rows": rows,
        "method_rows": method_rows,
        "core_rows": core_rows,
        "per_image": per_image,
        "summary": summary,
        "report_markdown": report_md,
        "report_path": str(report_path) if report_path is not None else None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "figures": figures,
        "visual_sections": visual_sections,
        "output_dir": str(run_dir) if run_dir is not None else None,
        "cache_root": str(cache_root),
    }


def render_alpha_segment_report(result, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_result = dict(result)
    summary = run_result["summary"]
    figures = _render_and_save_report_figures(
        run_dir=output_dir,
        summary=summary,
        rows=run_result["rows"],
        method_rows=run_result["method_rows"],
        method_specs=run_result["method_specs"],
    )
    visual_sections = _render_visual_sections(
        run_dir=output_dir,
        summary=summary,
        method_rows=run_result["method_rows"],
        visual_image_paths=summary["visual_image_paths"],
        method_specs=run_result["method_specs"],
    )
    report_md = _build_report_markdown(summary, figures=figures, visual_sections=visual_sections, method_specs=run_result["method_specs"])
    report_path = output_dir / DEFAULT_REPORT_FILENAME
    summary_path = output_dir / DEFAULT_SUMMARY_JSON
    report_path.write_text(report_md + "\n", encoding="utf-8")
    summary_path.write_text(
        _pretty_json(
            {
                "summary": summary,
                "rows": run_result["rows"],
                "method_rows": run_result["method_rows"],
                "core_rows": run_result["core_rows"],
                "visual_sections": visual_sections,
            }
        ),
        encoding="utf-8",
    )
    return {
        "report_markdown": report_md,
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "figures": figures,
        "visual_sections": visual_sections,
        "output_dir": str(output_dir),
    }


def compute_segment_conductance(
    model,
    hook,
    x,
    x0,
    target_class,
    *,
    n_steps=64,
    segment_start=0.0,
    segment_end=1.0,
    fd_eps=1e-3,
    clear_every=8,
    warn_on_fallback=False,
):
    x = x.contiguous()
    x0 = x0.contiguous()
    delta_x = (x - x0).contiguous()
    selected_steps = _segment_midpoint_mask(n_steps, segment_start, segment_end)
    if not selected_steps:
        raise ValueError(
            f"alpha segment [{segment_start}, {segment_end}] has no midpoint samples for n_steps={n_steps}"
        )

    def forward_with_layer(x_in):
        hook.clear()
        out = model(x_in)
        act = IG.unwrap_tensor(hook.get())
        return out, act

    alphas = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=x.device, dtype=x.dtype)
    step = 1.0 / float(n_steps)
    cond_accum = None
    used_fallback = False
    used_steps = 0

    for k in range(int(n_steps)):
        if k not in selected_steps:
            continue
        alpha = (alphas[k] + alphas[k + 1]) / 2.0
        x_alpha = (x0 + alpha * delta_x).contiguous().detach().requires_grad_(True)

        out, act = forward_with_layer(x_alpha)
        _, logits = IG.split_classifier_output(out)
        score = logits[0, target_class]
        grad_y = torch.autograd.grad(score, act, retain_graph=False, create_graph=False)[0]

        try:
            def act_only(inp):
                inp = inp.contiguous()
                _, activation = forward_with_layer(inp)
                return activation

            _, act_tangent = IG.jvp(act_only, (x_alpha,), (delta_x,))
        except RuntimeError as exc:
            if "view size is not compatible with input tensor's size and stride" not in str(exc):
                raise
            used_fallback = True
            alpha_plus = min(float(alpha.item()) + float(fd_eps), 1.0)
            alpha_minus = max(float(alpha.item()) - float(fd_eps), 0.0)
            denom = alpha_plus - alpha_minus
            if denom == 0.0:
                raise RuntimeError("Failed to build finite-difference fallback for dy/dalpha.")
            with torch.no_grad():
                _, act_plus = forward_with_layer((x0 + alpha_plus * delta_x).contiguous())
                _, act_minus = forward_with_layer((x0 + alpha_minus * delta_x).contiguous())
            act_tangent = (act_plus - act_minus) / denom
            del act_plus, act_minus

        integrand = grad_y * act_tangent
        cond_accum = integrand.detach() if cond_accum is None else (cond_accum + integrand.detach())
        used_steps += 1

        del x_alpha, out, act, logits, score, grad_y, act_tangent, integrand
        hook.clear()
        if clear_every > 0 and used_steps % int(clear_every) == 0:
            _clear_all_backend_caches()

    cond = cond_accum * step
    if used_fallback and warn_on_fallback:
        import warnings

        warnings.warn(
            "jvp on current backend failed; finite-difference fallback was used for dy/dalpha.",
            stacklevel=2,
        )
    return cond.detach(), int(used_steps)


def compute_segment_naa_attribution(
    model,
    hook,
    x,
    x0,
    target_class,
    *,
    n_steps=30,
    segment_start=0.0,
    segment_end=1.0,
    clear_every=8,
):
    x = x.contiguous()
    x0 = x0.contiguous()
    delta_x = (x - x0).contiguous()
    selected_indices = _segment_alpha_indices(n_steps, segment_start, segment_end)
    if not selected_indices:
        raise ValueError(f"alpha segment [{segment_start}, {segment_end}] has no samples for n_steps={n_steps}")

    def forward_with_layer(x_in):
        hook.clear()
        out = model(x_in)
        act = IG.unwrap_tensor(hook.get())
        return out, act

    with torch.no_grad():
        _, act_x0 = forward_with_layer(x0)
        x_end = (x0 + float(segment_end) * delta_x).contiguous()
        _, act_xe = forward_with_layer(x_end)
        delta_y = (act_xe - act_x0).detach()

    ia_accum = torch.zeros_like(delta_y)
    used_steps = 0
    for step_idx in selected_indices:
        alpha = float(step_idx + 1) / float(n_steps)
        x_alpha = (x0 + alpha * delta_x).contiguous().detach().requires_grad_(True)

        out, act = forward_with_layer(x_alpha)
        _, logits = IG.split_classifier_output(out)
        score = logits[0, target_class]
        grad_y = torch.autograd.grad(score, act, retain_graph=False, create_graph=False)[0]
        ia_accum = ia_accum + grad_y.detach()
        used_steps += 1

        del x_alpha, out, act, logits, score, grad_y
        hook.clear()
        if clear_every > 0 and used_steps % int(clear_every) == 0:
            _clear_all_backend_caches()

    ia = ia_accum / float(used_steps)
    attr = delta_y * ia
    return attr.detach(), ia.detach(), delta_y.detach(), int(used_steps)


def _require_scipy():
    if scipy_sparse is None or scipy_spsolve is None:
        raise ModuleNotFoundError(
            "Alpha-segment latent AOPC benchmark requires `scipy` for latent NLI imputation. "
            "Install it with `python3 -m pip install scipy` and rerun."
        ) from SCIPY_IMPORT_ERROR


def _load_core_record(*, image_path, layer_name, n_steps, blur_sigma, budget_percentiles, cache_root, refresh):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "blur_sigma": float(blur_sigma),
        "budget_percentiles": [int(v) for v in budget_percentiles],
        "schema": 1,
    }
    sidecar_key = _config_hash(config)
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="alpha_segment_classifier_core",
        config=config,
        label=Path(image_path).stem,
        compute_fn=lambda: _compute_classifier_core_value(
            image_path=image_path,
            layer_name=layer_name,
            blur_sigma=blur_sigma,
            budget_percentiles=budget_percentiles,
            cache_root=cache_root,
            sidecar_key=sidecar_key,
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
    method_spec,
    target_class_override=None,
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
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "method_spec": method_spec,
        "target_class_override": None if target_class_override is None else int(target_class_override),
        "top_n": int(top_n),
        "fd_eps": float(fd_eps),
        "clear_every": int(clear_every),
        "schema": 1,
    }
    sidecar_key = _config_hash(config)
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="alpha_segment_classifier_method",
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        compute_fn=lambda: _compute_classifier_method_value(
            image_path=image_path,
            layer_name=layer_name,
            n_steps=n_steps,
            method_spec=method_spec,
            target_class_override=target_class_override,
            cache_root=cache_root,
            sidecar_key=sidecar_key,
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
    n_steps,
    method_spec,
    core_value,
    method_value,
    donor_kind,
    budget_percentiles,
    blur_sigma,
    cache_root,
    refresh,
):
    image_key = image_signature(image_path)
    config = {
        "task": "classifier",
        "image": image_key,
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "method_spec": method_spec,
        "donor_kind": donor_kind,
        "budget_percentiles": [int(v) for v in budget_percentiles],
        "blur_sigma": float(blur_sigma),
        "target_class": int(core_value["target_class"]),
        "schema": 1,
    }
    return load_or_compute_cached_value(
        cache_root=cache_root,
        namespace="alpha_segment_classifier_evaluation",
        config=config,
        label=f"{Path(image_path).stem}_{method_spec['id']}_{donor_kind}",
        compute_fn=lambda: _compute_classifier_evaluation_value(
            image_path=image_path,
            layer_name=layer_name,
            core_value=core_value,
            method_value=method_value,
            donor_kind=donor_kind,
            budget_percentiles=budget_percentiles,
        ),
        refresh=refresh,
        required_device=None,
        current_device=current_device_label(getattr(IG, "DEVICE", None)),
    )


def _compute_classifier_core_value(*, image_path, layer_name, blur_sigma, budget_percentiles, cache_root, sidecar_key):
    _clear_all_backend_caches()
    x, image_np = IG.load_image(image_path)
    x_black = IG.black_baseline_like(x)
    x_blur, _ = build_image_baseline(x, image_np, mode="blur", blur_sigma=blur_sigma)

    hook = IG.LayerHook(IG.model, layer_name)
    try:
        def forward_with_layer(x_in):
            hook.clear()
            out = IG.model(x_in)
            act = IG.unwrap_tensor(hook.get())
            return out, act

        with torch.no_grad():
            clean_out, clean_act = forward_with_layer(x)
            black_out, black_act = forward_with_layer(x_black)
            blur_out, blur_act = forward_with_layer(x_blur)
            _, clean_logits = IG.split_classifier_output(clean_out)
            clean_probs = torch.softmax(clean_logits[0], dim=0)
            target_class = int(clean_logits[0].argmax().item())
            target_name = IG.class_names[target_class]
            clean_target_logit = float(clean_logits[0, target_class].item())
            clean_target_prob = float(clean_probs[target_class].item())
            _, black_logits = IG.split_classifier_output(black_out)
            _, blur_logits = IG.split_classifier_output(blur_out)
            black_target_logit = float(black_logits[0, target_class].item())
            blur_target_logit = float(blur_logits[0, target_class].item())

        clean_path = _array_sidecar_path(
            cache_root=cache_root,
            namespace="alpha_segment_classifier_core",
            label=Path(image_path).stem,
            sidecar_key=sidecar_key,
            kind="clean_act",
        )
        black_path = _array_sidecar_path(
            cache_root=cache_root,
            namespace="alpha_segment_classifier_core",
            label=Path(image_path).stem,
            sidecar_key=sidecar_key,
            kind="black_act",
        )
        blur_path = _array_sidecar_path(
            cache_root=cache_root,
            namespace="alpha_segment_classifier_core",
            label=Path(image_path).stem,
            sidecar_key=sidecar_key,
            kind="blur_act",
        )
        _save_array_sidecar(clean_path, clean_act.detach().cpu().numpy().astype(np.float32, copy=False))
        _save_array_sidecar(black_path, black_act.detach().cpu().numpy().astype(np.float32, copy=False))
        _save_array_sidecar(blur_path, blur_act.detach().cpu().numpy().astype(np.float32, copy=False))

        budget_counts = _budget_counts(clean_act.numel(), budget_percentiles)
        return {
            "task": "classifier",
            "image_path": str(image_path),
            "image_name": Path(image_path).name,
            "layer_name": str(layer_name),
            "target_class": int(target_class),
            "target_name": target_name,
            "clean_target_logit": float(clean_target_logit),
            "clean_target_prob": float(clean_target_prob),
            "black_target_logit": float(black_target_logit),
            "blur_target_logit": float(blur_target_logit),
            "image_height": int(image_np.shape[0]),
            "image_width": int(image_np.shape[1]),
            "n_units_total": int(clean_act.numel()),
            "activation_shape": [int(v) for v in clean_act.shape],
            "budget_percentiles": [int(v) for v in budget_percentiles],
            "budget_counts": [int(v) for v in budget_counts],
            "clean_act_path": str(clean_path),
            "black_act_path": str(black_path),
            "blur_act_path": str(blur_path),
            "blur_sigma": float(blur_sigma),
        }
    finally:
        hook.remove()
        _clear_all_backend_caches()


def _compute_classifier_method_value(
    *,
    image_path,
    layer_name,
    n_steps,
    method_spec,
    target_class_override,
    cache_root,
    sidecar_key,
    top_n,
    fd_eps,
    clear_every,
):
    _clear_all_backend_caches()
    payload = _run_classifier_method(
        image_path=image_path,
        layer_name=layer_name,
        n_steps=n_steps,
        method_spec=method_spec,
        target_class_override=target_class_override,
        top_n=top_n,
        fd_eps=fd_eps,
        clear_every=clear_every,
    )
    cond_tensor = payload["cond_tensor"]
    cond_np = cond_tensor[0].detach().cpu().numpy().astype(np.float32, copy=False)
    unit_scores = cond_np.reshape(-1).astype(np.float32, copy=False)
    overlay_map = _project_overlay_map(payload).astype(np.float32, copy=False)
    unit_scores_path = _array_sidecar_path(
        cache_root=cache_root,
        namespace="alpha_segment_classifier_method",
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        sidecar_key=sidecar_key,
        kind="unit_scores",
    )
    overlay_path = _array_sidecar_path(
        cache_root=cache_root,
        namespace="alpha_segment_classifier_method",
        label=f"{Path(image_path).stem}_{method_spec['id']}",
        sidecar_key=sidecar_key,
        kind="overlay_map",
    )
    _save_array_sidecar(unit_scores_path, unit_scores)
    _save_array_sidecar(overlay_path, overlay_map)
    _clear_all_backend_caches()
    return _serialize_method_payload(
        payload,
        method_spec,
        unit_scores_path=unit_scores_path,
        overlay_map_path=overlay_path,
        overlay_shape=overlay_map.shape,
    )


def _compute_classifier_evaluation_value(*, image_path, layer_name, core_value, method_value, donor_kind, budget_percentiles):
    _clear_all_backend_caches()
    runner = _ClassifierLatentAOPCRunner(
        image_path=image_path,
        layer_name=layer_name,
        core_value=core_value,
    )
    try:
        return _evaluate_method_with_runner(
            method_value=method_value,
            donor_kind=donor_kind,
            runner=runner,
            budget_percentiles=budget_percentiles,
        )
    finally:
        runner.close()
        _clear_all_backend_caches()


def _run_classifier_method(*, image_path, layer_name, n_steps, method_spec, target_class_override, top_n, fd_eps, clear_every):
    kind = method_spec["kind"]
    common = {
        "image_path": image_path,
        "layer_name": layer_name,
        "n_steps": n_steps,
        "baseline_mode": "zero",
        "top_n": top_n,
        "clear_every": clear_every,
        "verbose": False,
        "show_total_plot": False,
        "show_filter_plots": False,
        "target_class_override": target_class_override,
    }
    if kind == "ig":
        return _run_classifier_segment_ig_pipeline(
            segment_start=method_spec.get("segment_start", 0.0),
            segment_end=method_spec.get("segment_end", 1.0),
            fd_eps=fd_eps,
            **common,
        )
    if kind == "naa":
        return _run_classifier_segment_naa_pipeline(
            segment_start=method_spec.get("segment_start", 0.0),
            segment_end=method_spec.get("segment_end", 1.0),
            **common,
        )
    if kind == "cheap_ig":
        return cheap_ig.run_classifier_cheap_ig_pipeline(
            segment_start=method_spec.get("segment_start", 0.0),
            segment_end=method_spec.get("segment_end", 0.1),
            selection_mode=method_spec.get("selection_mode", "positive"),
            selection_top_k=method_spec.get("selection_top_k", 4000),
            fill_mode=method_spec.get("fill_mode", "naa_scaled"),
            fill_rho=method_spec.get("fill_rho", 0.6),
            **common,
        )
    raise ValueError(f"Unsupported classifier method kind: {kind}")


def _run_classifier_segment_ig_pipeline(
    *,
    image_path,
    layer_name,
    n_steps,
    segment_start,
    segment_end,
    baseline_mode,
    top_n,
    clear_every,
    fd_eps,
    target_class_override=None,
    **_,
):
    if baseline_mode != "zero":
        raise ValueError("Segment IG benchmark expects zero attribution baseline.")
    x, image_np = IG.load_image(image_path)
    x0 = IG.black_baseline_like(x)
    hook = IG.LayerHook(IG.model, layer_name)
    try:
        def forward_with_layer(x_in):
            hook.clear()
            out = IG.model(x_in)
            act = IG.unwrap_tensor(hook.get())
            return out, act

        with torch.no_grad():
            out, act = forward_with_layer(x)
            _, logits = IG.split_classifier_output(out)
            if target_class_override is None:
                target_class = int(logits[0].argmax().item())
            else:
                target_class = int(target_class_override)
            target_name = IG.class_names[target_class]
            target_logit = float(logits[0, target_class].item())
            target_prob = float(torch.softmax(logits[0], dim=0)[target_class].item())
            out_x0, _ = forward_with_layer(x0)
            _, logits_x0 = IG.split_classifier_output(out_x0)
            fx = float(logits[0, target_class].item())
            fx0 = float(logits_x0[0, target_class].item())

        cond_tensor, segment_steps = compute_segment_conductance(
            model=IG.model,
            hook=hook,
            x=x,
            x0=x0,
            target_class=target_class,
            n_steps=n_steps,
            segment_start=segment_start,
            segment_end=segment_end,
            fd_eps=fd_eps,
            clear_every=clear_every,
        )
        filter_scores = IG.reduce_filter_scores(cond_tensor)
        layer_score = cond_tensor.sum()
        topk = min(10, filter_scores.numel())
        top_vals, top_idx = torch.topk(filter_scores, k=topk)
        abs_error = abs((fx - fx0) - float(layer_score.item()))
        return {
            "image_path": image_path,
            "layer_name": layer_name,
            "target_class": target_class,
            "target_name": target_name,
            "target_logit": target_logit,
            "target_prob": target_prob,
            "activation_shape": tuple(act.shape),
            "cond_tensor": cond_tensor,
            "filter_scores": filter_scores,
            "layer_score": layer_score,
            "top_idx": top_idx,
            "top_vals": top_vals,
            "fx": fx,
            "fx0": fx0,
            "abs_error": abs_error,
            "baseline_mode": "zero",
            "baseline_rgb": None,
            "baseline_blur_sigma": None,
            "image_np": image_np,
            "segment_start": float(segment_start),
            "segment_end": float(segment_end),
            "segment_steps": int(segment_steps),
        }
    finally:
        hook.remove()
        _clear_all_backend_caches()


def _run_classifier_segment_naa_pipeline(
    *,
    image_path,
    layer_name,
    n_steps,
    segment_start,
    segment_end,
    baseline_mode,
    top_n,
    clear_every,
    target_class_override=None,
    **_,
):
    if baseline_mode != "zero":
        raise ValueError("Segment NAA benchmark expects zero attribution baseline.")
    x, image_np = IG.load_image(image_path)
    x0 = IG.black_baseline_like(x)
    hook = IG.LayerHook(IG.model, layer_name)
    try:
        def forward_with_layer(x_in):
            hook.clear()
            out = IG.model(x_in)
            act = IG.unwrap_tensor(hook.get())
            return out, act

        with torch.no_grad():
            out, act = forward_with_layer(x)
            _, logits = IG.split_classifier_output(out)
            if target_class_override is None:
                target_class = int(logits[0].argmax().item())
            else:
                target_class = int(target_class_override)
            target_name = IG.class_names[target_class]
            target_logit = float(logits[0, target_class].item())
            target_prob = float(torch.softmax(logits[0], dim=0)[target_class].item())
            out_x0, _ = forward_with_layer(x0)
            _, logits_x0 = IG.split_classifier_output(out_x0)
            fx = float(logits[0, target_class].item())
            fx0 = float(logits_x0[0, target_class].item())

        cond_tensor, ia_tensor, delta_y, segment_steps = compute_segment_naa_attribution(
            model=IG.model,
            hook=hook,
            x=x,
            x0=x0,
            target_class=target_class,
            n_steps=n_steps,
            segment_start=segment_start,
            segment_end=segment_end,
            clear_every=clear_every,
        )
        filter_scores = IG.reduce_filter_scores(cond_tensor)
        layer_score = cond_tensor.sum()
        topk = min(10, filter_scores.numel())
        top_vals, top_idx = torch.topk(filter_scores, k=topk)
        abs_error = abs((fx - fx0) - float(layer_score.item()))
        return {
            "image_path": image_path,
            "layer_name": layer_name,
            "target_class": target_class,
            "target_name": target_name,
            "target_logit": target_logit,
            "target_prob": target_prob,
            "activation_shape": tuple(act.shape),
            "cond_tensor": cond_tensor,
            "ia_tensor": ia_tensor,
            "delta_y": delta_y,
            "filter_scores": filter_scores,
            "layer_score": layer_score,
            "top_idx": top_idx,
            "top_vals": top_vals,
            "fx": fx,
            "fx0": fx0,
            "abs_error": abs_error,
            "baseline_mode": "zero",
            "baseline_rgb": None,
            "baseline_blur_sigma": None,
            "image_np": image_np,
            "segment_start": float(segment_start),
            "segment_end": float(segment_end),
            "segment_steps": int(segment_steps),
        }
    finally:
        hook.remove()
        _clear_all_backend_caches()


def _project_overlay_map(payload):
    cond_tensor = payload["cond_tensor"]
    cond_np = cond_tensor[0].detach().cpu().numpy()
    if cond_np.ndim < 3:
        raise ValueError(
            "Alpha-segment benchmark requires cond_tensor with shape [B, C, H, W]. "
            f"Got {tuple(cond_tensor.shape)}."
        )
    overlay = cond_np.sum(axis=0).astype(np.float32, copy=False)
    image_np = payload["image_np"]
    if tuple(overlay.shape) != tuple(image_np.shape[:2]):
        overlay = IG._resize_map_nearest(overlay, image_np.shape[:2]).astype(np.float32, copy=False)
    return overlay


def _serialize_method_payload(payload, method_spec, *, unit_scores_path, overlay_map_path, overlay_shape):
    value = {
        "method_name": method_spec["name"],
        "method_id": method_spec["id"],
        "family_name": method_spec["family_name"],
        "kind": method_spec["kind"],
        "segment_start": float(method_spec.get("segment_start", 0.0)),
        "segment_end": float(method_spec.get("segment_end", 1.0)),
        "target_class": int(payload.get("target_class", -1)),
        "target_name": payload.get("target_name"),
        "target_logit": float(payload.get("target_logit", float("nan"))),
        "target_prob": float(payload.get("target_prob", float("nan"))),
        "abs_error": float(payload.get("abs_error", float("nan"))),
        "fx": float(payload.get("fx", float("nan"))),
        "fx0": float(payload.get("fx0", float("nan"))),
        "unit_scores_path": str(unit_scores_path),
        "overlay_map_path": str(overlay_map_path),
        "overlay_shape": [int(v) for v in overlay_shape],
        "segment_steps": int(payload.get("segment_steps", 0)),
    }
    for optional_key in (
        "selection_mode",
        "selection_top_k",
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


class _ClassifierLatentAOPCRunner:
    def __init__(self, *, image_path, layer_name, core_value):
        self.image_path = str(image_path)
        self.layer_name = str(layer_name)
        self.core_value = dict(core_value)
        self.x, self.image_np = IG.load_image(self.image_path)
        self.target_class = int(core_value["target_class"])
        self.target_name = core_value["target_name"]
        self.clean_target_logit = float(core_value["clean_target_logit"])
        self.clean_act = _load_array_sidecar(core_value["clean_act_path"])
        self.black_act = _load_array_sidecar(core_value["black_act_path"])
        self.blur_act = _load_array_sidecar(core_value["blur_act_path"])
        self.activation_shape = tuple(int(v) for v in core_value["activation_shape"])
        if tuple(self.clean_act.shape) != self.activation_shape:
            raise ValueError(
                f"Clean activation shape mismatch: cached={self.clean_act.shape}, expected={self.activation_shape}"
            )
        self.clean_act_tensor = torch.from_numpy(self.clean_act).to(device=IG.DEVICE, dtype=IG.DTYPE)
        self.black_act_tensor = torch.from_numpy(self.black_act).to(device=IG.DEVICE, dtype=IG.DTYPE)
        self.blur_act_tensor = torch.from_numpy(self.blur_act).to(device=IG.DEVICE, dtype=IG.DTYPE)
        self._delta_cache = {}
        self._donor_cache = {
            "zero_baseline": torch.zeros_like(self.clean_act_tensor),
            "black_act": self.black_act_tensor,
            "blur_act": self.blur_act_tensor,
        }

    def score_drop_for_unit_indices(self, unit_indices, *, donor_kind):
        unit_indices = tuple(sorted(int(v) for v in unit_indices))
        cache_key = (str(donor_kind), unit_indices)
        if cache_key in self._delta_cache:
            return self._delta_cache[cache_key]

        donor = self._resolve_donor(unit_indices, donor_kind=donor_kind)
        modules = dict(IG.model.named_modules())
        index_tensor = torch.tensor(unit_indices, device=donor.device, dtype=torch.long)
        handle = modules[self.layer_name].register_forward_hook(
            lambda module, inp, out: _patch_raw_neurons(out, donor, index_tensor)
        )
        try:
            with torch.no_grad():
                out = IG.model(self.x)
                _, logits = IG.split_classifier_output(out)
                patched_target_logit = float(logits[0, self.target_class].item())
            drop = float(self.clean_target_logit - patched_target_logit)
        finally:
            handle.remove()
        self._delta_cache[cache_key] = drop
        return drop

    def _resolve_donor(self, unit_indices, *, donor_kind):
        if donor_kind in self._donor_cache and donor_kind in {"zero_baseline", "black_act", "blur_act"}:
            return self._donor_cache[donor_kind]

        donor_cache_key = (str(donor_kind), tuple(unit_indices))
        cached = self._donor_cache.get(donor_cache_key)
        if cached is not None:
            return cached

        if donor_kind == "layer_mean_exclusive":
            donor = _build_layer_mean_exclusive_donor(self.clean_act_tensor, unit_indices)
        elif donor_kind == "spatial_nli_same_channel":
            donor = _build_spatial_nli_same_channel_donor(self.clean_act_tensor, unit_indices)
        else:
            raise ValueError(f"Unsupported donor_kind: {donor_kind}")

        self._donor_cache[donor_cache_key] = donor
        return donor

    def close(self):
        pass


def _evaluate_method_with_runner(*, method_value, donor_kind, runner, budget_percentiles):
    unit_scores = _load_array_sidecar(method_value["unit_scores_path"]).reshape(-1)
    order = _stable_descending_order(unit_scores)
    budget_counts = _budget_counts(unit_scores.size, budget_percentiles)

    drop_curve = []
    normalized_curve = []
    for count in budget_counts:
        drop = float(runner.score_drop_for_unit_indices(order[: int(count)], donor_kind=donor_kind))
        drop_curve.append(drop)
        normalized_curve.append(drop / max(abs(runner.clean_target_logit), DEFAULT_EPS))

    aoc20 = float(np.mean(drop_curve)) if drop_curve else float("nan")
    aoc20_norm = float(np.mean(normalized_curve)) if normalized_curve else float("nan")
    head_n = min(20, int(order.size))
    return {
        "donor_kind": donor_kind,
        "budget_percentiles": [int(v) for v in budget_percentiles],
        "budget_counts": [int(v) for v in budget_counts],
        "drop_curve": [float(v) for v in drop_curve],
        "drop_curve_normalized": [float(v) for v in normalized_curve],
        "aoc20": aoc20,
        "aoc20_norm": aoc20_norm,
        "score": aoc20,
        "clean_target_logit": float(runner.clean_target_logit),
        "ordered_unit_indices_head": [int(v) for v in order[:head_n].tolist()],
        "abs_error": float(method_value.get("abs_error", float("nan"))),
        "fx": float(method_value.get("fx", float("nan"))),
        "fx0": float(method_value.get("fx0", float("nan"))),
        "selected_neurons": method_value.get("selected_neurons"),
    }


def _build_summary(
    *,
    rows,
    core_rows,
    method_rows,
    method_specs,
    layer_name,
    n_steps,
    budget_percentiles,
    donor_kinds,
    segment_end_values,
    blur_sigma,
    cache_root,
    single_image_path,
    visual_image_paths,
):
    family_names = _family_names(method_specs)
    by_family_donor_end = {}
    for row in rows:
        key = (row["family_name"], row["donor_kind"], float(row["segment_end"]))
        by_family_donor_end.setdefault(key, []).append(row)

    curve_summaries = {family: {} for family in family_names}
    peak_summary_rows = []
    best_donor_per_family = {}
    donor_rank_rows = []

    for family_name in family_names:
        donor_scores = []
        for donor_kind in donor_kinds:
            end_records = []
            aoc_means = []
            aoc_stds = []
            norm_means = []
            norm_stds = []
            for segment_end in segment_end_values:
                method_rows_subset = by_family_donor_end.get((family_name, donor_kind, float(segment_end)), [])
                aoc_stats = _stats_record([row["aoc20"] for row in method_rows_subset])
                norm_stats = _stats_record([row["aoc20_norm"] for row in method_rows_subset])
                end_records.append(
                    {
                        "segment_end": float(segment_end),
                        "aoc20": aoc_stats,
                        "aoc20_norm": norm_stats,
                    }
                )
                aoc_means.append(float(aoc_stats["mean"]))
                aoc_stds.append(float(aoc_stats["std"]))
                norm_means.append(float(norm_stats["mean"]))
                norm_stds.append(float(norm_stats["std"]))

            score_at_01 = _mean_for_segment(end_records, 0.1, key="aoc20")
            norm_score_at_01 = _mean_for_segment(end_records, 0.1, key="aoc20_norm")
            best_end = _best_segment_end(end_records, key="aoc20")
            other_values = [
                float(record["aoc20"]["mean"])
                for record in end_records
                if abs(float(record["segment_end"]) - 0.1) > 1e-9 and record["aoc20"]["mean"] == record["aoc20"]["mean"]
            ]
            peak_contrast = (
                float(score_at_01 - np.mean(np.asarray(other_values, dtype=np.float64)))
                if score_at_01 == score_at_01 and other_values
                else float("nan")
            )
            mean_over_all = _nanmean(aoc_means)
            norm_mean_over_all = _nanmean(norm_means)
            donor_row = {
                "family_name": family_name,
                "donor_kind": donor_kind,
                "score_at_0_1": float(score_at_01),
                "score_at_0_1_norm": float(norm_score_at_01),
                "best_end": float(best_end) if best_end == best_end else float("nan"),
                "peak_contrast": float(peak_contrast),
                "aoc20_mean_over_all_ends": float(mean_over_all),
                "aoc20_norm_mean_over_all_ends": float(norm_mean_over_all),
            }
            donor_rank_rows.append(donor_row)
            donor_scores.append(donor_row)
            peak_summary_rows.append(donor_row)
            curve_summaries[family_name][donor_kind] = {
                "segment_end_values": [float(v) for v in segment_end_values],
                "aoc20_mean": [float(v) for v in aoc_means],
                "aoc20_std": [float(v) for v in aoc_stds],
                "aoc20_norm_mean": [float(v) for v in norm_means],
                "aoc20_norm_std": [float(v) for v in norm_stds],
                "score_at_0_1": float(score_at_01),
                "best_end": float(best_end) if best_end == best_end else float("nan"),
                "peak_contrast": float(peak_contrast),
                "aoc20_mean_over_all_ends": float(mean_over_all),
                "aoc20_norm_mean_over_all_ends": float(norm_mean_over_all),
            }

        best_donor_per_family[family_name] = _best_donor_name(donor_scores)

    method_core_stats = {}
    by_method_family_end = {}
    for row in method_rows:
        key = (row["family_name"], float(row["segment_end"]))
        by_method_family_end.setdefault(key, []).append(row)
    for family_name in family_names:
        method_core_stats[family_name] = []
        for segment_end in segment_end_values:
            family_rows = by_method_family_end.get((family_name, float(segment_end)), [])
            method_core_stats[family_name].append(
                {
                    "segment_end": float(segment_end),
                    "runtime_s": _stats_record([row["method_duration_s"] for row in family_rows]),
                    "abs_error": _stats_record([row.get("abs_error") for row in family_rows]),
                    "segment_steps": _stats_record([row.get("segment_steps") for row in family_rows]),
                }
            )

    return {
        "task": "classifier",
        "layer_name": str(layer_name),
        "n_steps": int(n_steps),
        "budget_percentiles": [int(v) for v in budget_percentiles],
        "segment_end_values": [float(v) for v in segment_end_values],
        "donor_kinds": list(donor_kinds),
        "blur_sigma": float(blur_sigma),
        "n_images": int(len(core_rows)),
        "cache_root": str(cache_root),
        "single_image_path": single_image_path,
        "single_image_name": Path(single_image_path).name if single_image_path else None,
        "visual_image_paths": [str(path) for path in visual_image_paths],
        "family_names": family_names,
        "best_donor_per_family": best_donor_per_family,
        "curve_summaries": curve_summaries,
        "peak_summary_rows": peak_summary_rows,
        "method_core_stats": method_core_stats,
        "core_summary": {
            "n_units_total": _stats_record([row["core"]["n_units_total"] for row in core_rows]),
            "core_runtime_s": _stats_record([row["core_duration_s"] for row in core_rows]),
        },
        "donor_rank_rows": donor_rank_rows,
    }


def _render_and_save_report_figures(*, run_dir, summary, rows, method_rows, method_specs):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for family_name in summary["family_names"]:
        slug = _safe_slug(family_name)
        figures[f"aggregate_{slug}"] = _save_figure(
            _plot_aggregate_curves(summary, family_name, normalized=False),
            figure_dir / f"{slug}_aggregate_curves.png",
        )
        figures[f"aggregate_norm_{slug}"] = _save_figure(
            _plot_aggregate_curves(summary, family_name, normalized=True),
            figure_dir / f"{slug}_aggregate_curves_normalized.png",
        )
        figures[f"single_{slug}"] = _save_figure(
            _plot_single_image_curves(summary, rows, family_name),
            figure_dir / f"{slug}_single_image_curves.png",
        )
        figures[f"heatmap_{slug}"] = _save_figure(
            _plot_donor_segment_heatmap(summary, family_name),
            figure_dir / f"{slug}_donor_vs_segment_heatmap.png",
        )
    figures["peak_summary"] = _save_figure(
        _plot_peak_summary(summary),
        figure_dir / "peak_summary.png",
    )
    return figures


def _render_visual_sections(*, run_dir, summary, method_rows, visual_image_paths, method_specs):
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    family_names = summary["family_names"]
    segment_end_values = summary["segment_end_values"]
    method_index = {
        (row["image_path"], row["family_name"], float(row["segment_end"])): row
        for row in method_rows
    }

    for family_name in family_names:
        best_donor = summary["best_donor_per_family"].get(family_name)
        for page_idx, image_chunk in enumerate(_chunked(visual_image_paths, DEFAULT_VISUAL_PAGE_SIZE), start=1):
            fig = _build_visual_composite_figure(
                family_name=family_name,
                image_paths=image_chunk,
                segment_end_values=segment_end_values,
                method_index=method_index,
                best_donor=best_donor,
            )
            key = f"{_safe_slug(family_name)}_visual_page_{page_idx}"
            figure_path = figure_dir / f"{key}.png"
            saved_path = _save_figure(fig, figure_path)
            sections.append(
                {
                    "family_name": family_name,
                    "page_idx": int(page_idx),
                    "best_donor": best_donor,
                    "figure_path": saved_path,
                }
            )
    return sections


def _plot_aggregate_curves(summary, family_name, *, normalized):
    donor_order = summary["donor_kinds"]
    family_curves = summary["curve_summaries"][family_name]
    x = np.asarray(summary["segment_end_values"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    key_mean = "aoc20_norm_mean" if normalized else "aoc20_mean"
    key_std = "aoc20_norm_std" if normalized else "aoc20_std"
    ylabel = "Mean AOC20 / |clean logit|" if normalized else "Mean AOC20"
    title_suffix = "normalized" if normalized else "raw"
    for donor_kind in donor_order:
        record = family_curves[donor_kind]
        y = np.asarray(record[key_mean], dtype=np.float64)
        std = np.asarray(record[key_std], dtype=np.float64)
        color = _donor_color(donor_kind)
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=_donor_label(donor_kind))
        lower = y - std
        upper = y + std
        ax.fill_between(x, lower, upper, color=color, alpha=0.18)
    ax.set_xticks(x)
    ax.set_xlabel("Segment end")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{family_name}: aggregate latent AOPC curves ({title_suffix})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return fig


def _plot_single_image_curves(summary, rows, family_name):
    x = np.asarray(summary["segment_end_values"], dtype=np.float64)
    image_path = summary["single_image_path"]
    donor_order = summary["donor_kinds"]
    fig, ax = plt.subplots(figsize=(9.5, 5.4), constrained_layout=True)
    for donor_kind in donor_order:
        y = []
        for segment_end in x.tolist():
            value = float("nan")
            for row in rows:
                if (
                    row["image_path"] == image_path
                    and row["family_name"] == family_name
                    and row["donor_kind"] == donor_kind
                    and abs(float(row["segment_end"]) - float(segment_end)) <= 1e-9
                ):
                    value = float(row["aoc20"])
                    break
            y.append(value)
        color = _donor_color(donor_kind)
        ax.plot(x, np.asarray(y, dtype=np.float64), marker="o", linewidth=2.0, color=color, label=_donor_label(donor_kind))
    ax.set_xticks(x)
    ax.set_xlabel("Segment end")
    ax.set_ylabel("AOC20")
    ax.set_title(f"{family_name}: single-image latent AOPC ({summary['single_image_name']})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    return fig


def _plot_donor_segment_heatmap(summary, family_name):
    donor_order = summary["donor_kinds"]
    x_values = summary["segment_end_values"]
    matrix = np.full((len(donor_order), len(x_values)), np.nan, dtype=np.float64)
    for row_idx, donor_kind in enumerate(donor_order):
        record = summary["curve_summaries"][family_name][donor_kind]
        matrix[row_idx, :] = np.asarray(record["aoc20_mean"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(10.0, 3.4), constrained_layout=True)
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{value:.1f}" for value in x_values])
    ax.set_yticks(np.arange(len(donor_order)))
    ax.set_yticklabels([_donor_label(kind) for kind in donor_order])
    ax.set_xlabel("Segment end")
    ax.set_title(f"{family_name}: donor vs segment mean AOC20")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            ax.text(col_idx, row_idx, "n/a" if value != value else f"{value:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean AOC20")
    return fig


def _plot_peak_summary(summary):
    rows = list(summary["peak_summary_rows"])
    rows.sort(key=lambda row: (row["family_name"], -row["aoc20_mean_over_all_ends"], row["donor_kind"]))
    labels = [f"{row['family_name']} / {_donor_label(row['donor_kind'])}" for row in rows]
    values = [float(row["aoc20_mean_over_all_ends"]) for row in rows]
    score01 = [float(row["score_at_0_1"]) for row in rows]
    colors = [_donor_color(row["donor_kind"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12.0, max(4.0, 0.45 * len(rows) + 1.5)), constrained_layout=True)
    y = np.arange(len(rows))
    ax.barh(y, values, color=colors, alpha=0.7, label="Mean over all ends")
    ax.scatter(score01, y, color="black", marker="D", s=35, label="Score@0.1")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("AOC20")
    ax.set_title("Peak summary by method family and donor")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="best")
    return fig


def _build_visual_composite_figure(*, family_name, image_paths, segment_end_values, method_index, best_donor):
    n_rows = len(image_paths)
    n_cols = len(segment_end_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols, 2.4 * n_rows + 0.7),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(f"{family_name}: attribution overlays by segment end | best donor={_donor_label(best_donor)}", fontsize=14)

    for row_idx, image_path in enumerate(image_paths):
        _, image_np = IG.load_image(str(image_path))
        for col_idx, segment_end in enumerate(segment_end_values):
            ax = axes[row_idx, col_idx]
            record = method_index.get((str(image_path), family_name, float(segment_end)))
            if record is None:
                ax.set_axis_off()
                continue
            overlay = _load_array_sidecar(record["overlay_map_path"])
            overlay = _normalize_map(overlay)
            if tuple(overlay.shape) != tuple(image_np.shape[:2]):
                overlay = IG._resize_map_nearest(overlay, image_np.shape[:2]).astype(np.float32, copy=False)
            ax.imshow(image_np, interpolation="nearest")
            ax.imshow(overlay, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"e={float(segment_end):.1f}", fontsize=9)
            if col_idx == 0:
                ax.text(
                    -0.02,
                    0.5,
                    Path(image_path).name,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=8,
                    rotation=90,
                )
    return fig


def _build_report_markdown(summary, *, figures, visual_sections, method_specs):
    lines = [
        "# Alpha-Segment Latent AOPC Sweep",
        "",
        "Classifier-only benchmark for `yolo11s-cls` with raw-neuron deletion AOC20 in latent space.",
        "",
        "## Configuration",
        "",
        f"- layer_name=`{summary['layer_name']}`",
        f"- n_steps=`{summary['n_steps']}`",
        f"- budget_percentiles=`{summary['budget_percentiles']}`",
        f"- segment_end_values=`{summary['segment_end_values']}`",
        f"- donor_kinds=`{summary['donor_kinds']}`",
        f"- blur_sigma=`{summary['blur_sigma']}`",
        f"- n_images=`{summary['n_images']}`",
        "",
        "## Peak Summary",
        "",
        _build_peak_summary_markdown_table(summary["peak_summary_rows"]),
        "",
    ]

    peak_figure = figures.get("peak_summary")
    if peak_figure:
        lines.extend([f"![]({_relative_markdown_path(peak_figure)})", ""])

    for family_name in summary["family_names"]:
        slug = _safe_slug(family_name)
        lines.extend(
            [
                f"## {family_name}",
                "",
                f"- best donor by aggregate metric: `{_donor_label(summary['best_donor_per_family'][family_name])}`",
                "",
            ]
        )
        for key in (
            f"aggregate_{slug}",
            f"aggregate_norm_{slug}",
            f"single_{slug}",
            f"heatmap_{slug}",
        ):
            path = figures.get(key)
            if path:
                lines.extend([f"![]({_relative_markdown_path(path)})", ""])

        family_sections = [section for section in visual_sections if section["family_name"] == family_name]
        if family_sections:
            lines.extend(["### Visual Tables", ""])
            for section in family_sections:
                lines.append(
                    f"Page {section['page_idx']} | best donor `{_donor_label(section['best_donor'])}`"
                )
                lines.append("")
                lines.append(f"![]({_relative_markdown_path(section['figure_path'])})")
                lines.append("")
    return "\n".join(lines)


def _build_peak_summary_markdown_table(rows):
    ordered = sorted(rows, key=lambda row: (row["family_name"], row["donor_kind"]))
    lines = [
        "| Method | Donor | score@0.1 | best_end | peak_contrast | AOC20 mean(all ends) | AOC20 norm mean(all ends) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        lines.append(
            "| {family} | {donor} | {score_01} | {best_end} | {peak} | {mean_all} | {mean_norm_all} |".format(
                family=row["family_name"],
                donor=_donor_label(row["donor_kind"]),
                score_01=_format_number(row["score_at_0_1"]),
                best_end=_format_number(row["best_end"], digits=2),
                peak=_format_number(row["peak_contrast"]),
                mean_all=_format_number(row["aoc20_mean_over_all_ends"]),
                mean_norm_all=_format_number(row["aoc20_norm_mean_over_all_ends"]),
            )
        )
    return "\n".join(lines)


def _patch_raw_neurons(out, donor, index_tensor):
    if not torch.is_tensor(out):
        raise TypeError(f"Expected tensor output for patch hook, got {type(out).__name__}")
    patched = out.clone()
    flat_patched = patched.reshape(patched.shape[0], -1)
    flat_donor = donor.reshape(donor.shape[0], -1)
    flat_patched[:, index_tensor] = flat_donor[:, index_tensor]
    return flat_patched.reshape_as(patched)


def _build_layer_mean_exclusive_donor(clean_act, unit_indices):
    donor = clean_act.clone()
    flat = donor.reshape(donor.shape[0], -1)
    clean_flat = clean_act.reshape(clean_act.shape[0], -1)
    mask = torch.zeros(flat.shape[1], device=flat.device, dtype=torch.bool)
    if unit_indices:
        mask[torch.tensor(unit_indices, device=flat.device, dtype=torch.long)] = True
    if torch.any(~mask):
        mean_value = clean_flat[:, ~mask].mean()
    else:
        mean_value = clean_flat.mean()
    flat[:, mask] = mean_value
    return flat.reshape_as(donor)


def _build_spatial_nli_same_channel_donor(clean_act, unit_indices):
    if clean_act.ndim != 4:
        raise ValueError(
            "spatial_nli_same_channel donor requires activation shape [B, C, H, W], "
            f"got {tuple(clean_act.shape)}"
        )
    if clean_act.shape[0] != 1:
        raise ValueError(f"Expected batch size 1 for donor construction, got shape={tuple(clean_act.shape)}")

    donor_np = clean_act.detach().cpu().numpy().astype(np.float32, copy=True)
    _, channels, height, width = donor_np.shape
    flat_mask = np.zeros(channels * height * width, dtype=bool)
    if unit_indices:
        flat_mask[np.asarray(unit_indices, dtype=np.int64)] = True
    mask_by_channel = flat_mask.reshape(channels, height, width)

    for channel_idx in range(channels):
        channel_mask = mask_by_channel[channel_idx]
        if not np.any(channel_mask):
            continue
        channel_values = donor_np[0, channel_idx]
        solved = _impute_single_channel(channel_values, channel_mask)
        donor_np[0, channel_idx, channel_mask] = solved[channel_mask]
    return torch.from_numpy(donor_np).to(device=clean_act.device, dtype=clean_act.dtype)


def _impute_single_channel(channel_values, channel_mask):
    channel_values = np.asarray(channel_values, dtype=np.float64)
    channel_mask = np.asarray(channel_mask, dtype=bool)
    if not np.any(channel_mask):
        return channel_values.astype(np.float32, copy=True)

    matrix, rhs, missing_indices = _build_sparse_system_2d(channel_mask, channel_values)
    if matrix.shape[0] == 0:
        return _channel_mean_fill(channel_values, channel_mask)

    try:
        solved = scipy_spsolve(matrix, rhs)
    except Exception:
        return _channel_mean_fill(channel_values, channel_mask)
    solved = np.asarray(solved, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(solved)):
        return _channel_mean_fill(channel_values, channel_mask)

    out = channel_values.reshape(-1).copy()
    out[missing_indices] = solved
    return out.reshape(channel_values.shape).astype(np.float32, copy=False)


def _channel_mean_fill(channel_values, channel_mask):
    arr = np.asarray(channel_values, dtype=np.float64)
    mask = np.asarray(channel_mask, dtype=bool)
    if np.any(~mask):
        fill = float(arr[~mask].mean())
    else:
        fill = float(arr.mean())
    out = arr.copy()
    out[mask] = fill
    return out.astype(np.float32, copy=False)


def _build_sparse_system_2d(mask_2d, values_2d):
    height, width = mask_2d.shape
    flat_mask = mask_2d.reshape(-1)
    missing_indices = np.flatnonzero(flat_mask)
    coords_to_row = np.full(flat_mask.shape[0], -1, dtype=np.int64)
    coords_to_row[missing_indices] = np.arange(missing_indices.size, dtype=np.int64)
    value_flat = values_2d.reshape(-1)

    rows = []
    cols = []
    data = []
    rhs = np.zeros(missing_indices.size, dtype=np.float64)

    for row_idx, flat_idx in enumerate(missing_indices.tolist()):
        y, x = divmod(int(flat_idx), int(width))
        diag = 0.0
        for (dy, dx), weight in NLI_NEIGHBOR_WEIGHTS:
            ny = y + int(dy)
            nx = x + int(dx)
            if ny < 0 or ny >= height or nx < 0 or nx >= width:
                continue
            neighbor_flat = int(ny * width + nx)
            diag += float(weight)
            if flat_mask[neighbor_flat]:
                rows.append(row_idx)
                cols.append(int(coords_to_row[neighbor_flat]))
                data.append(-float(weight))
            else:
                rhs[row_idx] += float(weight) * float(value_flat[neighbor_flat])
        if diag <= DEFAULT_EPS:
            rows.append(row_idx)
            cols.append(row_idx)
            data.append(1.0)
            rhs[row_idx] = float(value_flat[flat_idx])
        else:
            rows.append(row_idx)
            cols.append(row_idx)
            data.append(float(diag))

    matrix = scipy_sparse.csr_matrix(
        (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
        shape=(missing_indices.size, missing_indices.size),
    )
    return matrix, rhs, missing_indices


def _segment_midpoint_mask(n_steps, start, end):
    start = float(start)
    end = float(end)
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError(f"Expected 0 <= start <= end <= 1, got [{start}, {end}]")
    mids = (np.arange(int(n_steps), dtype=np.float64) + 0.5) / float(n_steps)
    if end < 1.0:
        chosen = np.nonzero((mids >= start) & (mids < end))[0]
    else:
        chosen = np.nonzero((mids >= start) & (mids <= end))[0]
    return {int(idx) for idx in chosen.tolist()}


def _segment_alpha_indices(n_steps, start, end):
    start = float(start)
    end = float(end)
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError(f"Expected 0 <= start <= end <= 1, got [{start}, {end}]")
    alphas = np.linspace(1.0 / float(n_steps), 1.0, int(n_steps), dtype=np.float64)
    if end < 1.0:
        chosen = np.nonzero((alphas >= start) & (alphas < end))[0]
    else:
        chosen = np.nonzero((alphas >= start) & (alphas <= end))[0]
    return [int(idx) for idx in chosen.tolist()]


def _budget_counts(n_units, budget_percentiles):
    counts = []
    for percentile in budget_percentiles:
        counts.append(max(1, min(int(n_units), int(np.ceil(float(percentile) / 100.0 * float(n_units))))))
    return [int(v) for v in counts]


def _family_names(method_specs):
    ordered = []
    for spec in method_specs:
        family_name = str(spec["family_name"])
        if family_name not in ordered:
            ordered.append(family_name)
    return ordered


def _mean_for_segment(end_records, segment_end, *, key):
    for record in end_records:
        if abs(float(record["segment_end"]) - float(segment_end)) <= 1e-9:
            return float(record[key]["mean"])
    return float("nan")


def _best_segment_end(end_records, *, key):
    best_value = float("-inf")
    best_end = float("nan")
    for record in end_records:
        value = float(record[key]["mean"])
        if value != value:
            continue
        segment_end = float(record["segment_end"])
        if value > best_value + 1e-12 or (abs(value - best_value) <= 1e-12 and segment_end < best_end):
            best_value = value
            best_end = segment_end
    return best_end


def _best_donor_name(donor_rows):
    ordered = sorted(
        donor_rows,
        key=lambda row: (
            -float(row["aoc20_mean_over_all_ends"]) if row["aoc20_mean_over_all_ends"] == row["aoc20_mean_over_all_ends"] else float("inf"),
            -float(row["score_at_0_1"]) if row["score_at_0_1"] == row["score_at_0_1"] else float("inf"),
            _donor_order().index(row["donor_kind"]) if row["donor_kind"] in _donor_order() else len(_donor_order()),
        ),
    )
    return ordered[0]["donor_kind"] if ordered else None


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


def _nanmean(values):
    array = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    valid = array[np.isfinite(array)]
    return float(valid.mean()) if valid.size else float("nan")


def _normalize_donor_kinds(donor_kinds):
    normalized = []
    for donor_kind in donor_kinds:
        donor_kind = str(donor_kind)
        if donor_kind not in DEFAULT_DONOR_KINDS:
            raise ValueError(
                f"Unsupported donor_kind: {donor_kind}. Supported values: {list(DEFAULT_DONOR_KINDS)}"
            )
        if donor_kind not in normalized:
            normalized.append(donor_kind)
    return tuple(normalized)


def _normalize_budget_percentiles(percentiles):
    values = [int(value) for value in percentiles]
    if not values:
        raise ValueError("budget_percentiles must not be empty.")
    for value in values:
        if value < 1 or value > 100:
            raise ValueError(f"Each budget percentile must be in [1, 100], got {value}.")
    return tuple(values)


def _normalize_method_specs(method_specs):
    normalized = []
    seen_ids = set()
    for raw_spec in method_specs:
        spec = dict(raw_spec)
        spec["kind"] = str(spec["kind"])
        spec["family_name"] = str(spec.get("family_name") or _default_family_name(spec["kind"]))
        spec["name"] = str(spec.get("name") or _default_method_name(spec))
        spec["segment_start"] = float(spec.get("segment_start", 0.0))
        spec["segment_end"] = float(spec.get("segment_end", 1.0))
        spec["id"] = _method_id(spec)
        if spec["id"] in seen_ids:
            raise ValueError(f"Duplicate method spec id: {spec['id']}. Please use unique names/params.")
        seen_ids.add(spec["id"])
        normalized.append(spec)
    return normalized


def _default_family_name(kind):
    mapping = {
        "ig": "IG",
        "naa": "NAA",
        "cheap_ig": "Cheap-IG",
    }
    return mapping.get(str(kind), str(kind))


def _default_method_name(spec):
    kind = str(spec["kind"])
    segment_start = float(spec.get("segment_start", 0.0))
    segment_end = float(spec.get("segment_end", 1.0))
    if kind == "ig":
        return f"IG[{segment_start:g},{segment_end:g}]"
    if kind == "naa":
        return f"NAA[{segment_start:g},{segment_end:g}]"
    selection_mode = spec.get("selection_mode", "positive")
    selection_top_k = int(spec.get("selection_top_k", 4000))
    fill_mode = str(spec.get("fill_mode", "naa_scaled"))
    if fill_mode == "zero":
        fill_fragment = "/fill-zero"
    else:
        fill_fragment = f"/fill-{fill_mode}-rho{float(spec.get('fill_rho', 0.6)):g}"
    return f"cheap-ig[{segment_start:g},{segment_end:g}]/{selection_mode}/k{selection_top_k}{fill_fragment}"


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


def _stable_descending_order(scores):
    values = np.asarray(scores, dtype=np.float64).copy()
    idx = np.arange(values.size, dtype=np.int64)
    values[~np.isfinite(values)] = -np.inf
    order = np.lexsort((idx, -values))
    return order.astype(np.int64, copy=False)


def _donor_order():
    return list(DEFAULT_DONOR_KINDS)


def _donor_label(donor_kind):
    mapping = {
        "zero_baseline": "zero_baseline",
        "black_act": "black_act",
        "blur_act": "blur_act",
        "layer_mean_exclusive": "layer_mean_exclusive",
        "spatial_nli_same_channel": "spatial_nli_same_channel",
    }
    return mapping.get(str(donor_kind), str(donor_kind))


def _donor_color(donor_kind):
    mapping = {
        "zero_baseline": "#4c566a",
        "black_act": "tab:blue",
        "blur_act": "tab:orange",
        "layer_mean_exclusive": "tab:green",
        "spatial_nli_same_channel": "tab:red",
    }
    return mapping.get(str(donor_kind), "tab:purple")


def _format_number(value, digits=4):
    if value is None:
        return "n/a"
    value = float(value)
    if value != value:
        return "n/a"
    return f"{value:.{digits}f}"


def _normalize_map(arr):
    arr = np.asarray(arr, dtype=np.float32)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    if max_abs > 0.0:
        arr = arr / max_abs
    return arr


def _config_hash(config):
    return hashlib.md5(json.dumps(_normalize_for_json(config), sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()[:12]


def _normalize_for_json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]
    return value


def _array_sidecar_path(*, cache_root, namespace, label, sidecar_key, kind):
    root = Path(cache_root)
    filename = f"{_safe_slug(label)}_{_safe_slug(kind)}_{sidecar_key}.npz"
    return root / namespace / "sidecars" / filename


def _save_array_sidecar(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, array=np.asarray(array))
    return str(path)


def _load_array_sidecar(path):
    payload = np.load(Path(path))
    return np.asarray(payload["array"])


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
    return str(path)


def _relative_markdown_path(path):
    path = Path(path)
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _pretty_json(payload):
    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True)


def _chunked(values, chunk_size):
    values = list(values)
    for start in range(0, len(values), int(chunk_size)):
        yield values[start:start + int(chunk_size)]
