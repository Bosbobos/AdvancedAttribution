import numpy as np
import torch

from modules import IG, IG_det
from modules.baseline_utils import DEFAULT_BLUR_SIGMA, build_image_baseline, baseline_title_fragment


def _format_float(value):
    return f"{float(value):.6g}"


def _alpha_grid(n_steps):
    n_steps = int(n_steps)
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    return np.linspace(1.0 / n_steps, 1.0, n_steps, dtype=np.float64)


def _segment_index_mask(alphas, start, end):
    start = float(start)
    end = float(end)
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError(f"alpha segment must satisfy 0 <= start <= end <= 1, got [{start}, {end}]")
    alphas = np.asarray(alphas, dtype=np.float64)
    if end < 1.0:
        return np.nonzero((alphas >= start) & (alphas < end))[0]
    return np.nonzero((alphas >= start) & (alphas <= end))[0]


def _selection_mask(scores, selection_mode, selection_top_k):
    scores = np.asarray(scores, dtype=np.float64)
    mask = np.zeros(scores.shape[0], dtype=bool)
    if selection_top_k <= 0 or scores.size == 0:
        return mask

    selection_top_k = int(min(selection_top_k, scores.size))

    if selection_mode == "signed":
        pos_idx = np.flatnonzero(scores > 0)
        neg_idx = np.flatnonzero(scores < 0)

        if pos_idx.size > 0:
            k_pos = min(selection_top_k, pos_idx.size)
            if k_pos >= pos_idx.size:
                mask[pos_idx] = True
            else:
                chosen_pos = np.argpartition(scores[pos_idx], pos_idx.size - k_pos)[-k_pos:]
                mask[pos_idx[chosen_pos]] = True

        if neg_idx.size > 0:
            k_neg = min(selection_top_k, neg_idx.size)
            if k_neg >= neg_idx.size:
                mask[neg_idx] = True
            else:
                chosen_neg = np.argpartition(np.abs(scores[neg_idx]), neg_idx.size - k_neg)[-k_neg:]
                mask[neg_idx[chosen_neg]] = True
    elif selection_mode == "unsigned":
        if selection_top_k >= scores.size:
            mask[:] = True
        else:
            chosen = np.argpartition(np.abs(scores), scores.size - selection_top_k)[-selection_top_k:]
            mask[chosen] = True
    elif selection_mode == "positive":
        pos_idx = np.flatnonzero(scores > 0)
        if pos_idx.size == 0:
            return mask
        k_pos = min(selection_top_k, pos_idx.size)
        if k_pos >= pos_idx.size:
            mask[pos_idx] = True
        else:
            chosen_pos = np.argpartition(scores[pos_idx], pos_idx.size - k_pos)[-k_pos:]
            mask[pos_idx[chosen_pos]] = True
    else:
        raise ValueError(f"Unsupported selection_mode: {selection_mode}")

    return mask


def _validate_segment(n_steps, segment_start, segment_end):
    segment_idx = _segment_index_mask(_alpha_grid(n_steps), segment_start, segment_end)
    if len(segment_idx) == 0:
        raise ValueError(
            f"alpha segment [{segment_start}, {segment_end}] has no samples for n_steps={n_steps}; "
            "increase n_steps or widen the segment"
        )


def _selection_title_fragment(selection_mode, selection_top_k):
    if selection_mode == "signed":
        return f"selection=signed, top_k_per_sign={int(selection_top_k)}"
    if selection_mode == "positive":
        return f"selection=positive, top_k={int(selection_top_k)}"
    return f"selection={selection_mode}, top_k={int(selection_top_k)}"


def _normalize_fill_mode(fill_mode):
    fill_mode = str(fill_mode).strip().lower()
    if fill_mode not in {"zero", "naa_scaled"}:
        raise ValueError(f"Unsupported fill_mode: {fill_mode}")
    return fill_mode


def _fill_title_fragment(fill_mode, fill_rho):
    fill_mode = _normalize_fill_mode(fill_mode)
    if fill_mode == "zero":
        return "fill=zero"
    return f"fill=naa_scaled,rho={_format_float(fill_rho)}"


def _scale_basis(values, selection_mode):
    arr = np.asarray(values, dtype=np.float64)
    if selection_mode == "positive":
        return np.maximum(arr, 0.0)
    return np.abs(arr)


def _residual_fill_values(values, selection_mode):
    arr = np.asarray(values, dtype=np.float64)
    if selection_mode == "positive":
        return np.maximum(arr, 0.0)
    return arr


def _hybrid_fill_vector(
    exact_vector,
    approx_vector,
    selected_mask,
    *,
    selection_mode,
    fill_mode,
    fill_rho,
    cap_quantile=0.1,
    eps=1e-12,
):
    fill_mode = _normalize_fill_mode(fill_mode)
    exact = np.asarray(exact_vector, dtype=np.float64)
    approx = np.asarray(approx_vector, dtype=np.float64)
    mask = np.asarray(selected_mask, dtype=bool)

    filled = np.zeros_like(exact, dtype=np.float64)
    filled[mask] = exact[mask]

    stats = {
        "fill_mode": fill_mode,
        "fill_rho": float(fill_rho),
        "fill_beta": 0.0,
        "filled_neurons": 0,
    }

    if fill_mode == "zero" or exact.size == 0 or (~mask).sum() == 0:
        return filled, stats

    exact_scale = _scale_basis(exact[mask], selection_mode)
    approx_scale_selected = _scale_basis(approx[mask], selection_mode)
    valid = (
        np.isfinite(exact_scale)
        & np.isfinite(approx_scale_selected)
        & (exact_scale > eps)
        & (approx_scale_selected > eps)
    )
    if not np.any(valid):
        return filled, stats

    beta_raw = float(np.median(exact_scale[valid] / (approx_scale_selected[valid] + eps)))
    residual_values = _residual_fill_values(approx, selection_mode)
    residual_scale = _scale_basis(approx[~mask], selection_mode)
    residual_scale = residual_scale[np.isfinite(residual_scale) & (residual_scale > eps)]

    if residual_scale.size == 0:
        stats["fill_beta"] = max(0.0, beta_raw)
        return filled, stats

    q_in = float(np.quantile(exact_scale[valid], float(cap_quantile)))
    max_out = float(residual_scale.max())
    beta_cap = float(fill_rho) * q_in / (max_out + eps) if q_in > eps else 0.0
    beta = float(min(beta_raw, beta_cap))
    beta = max(0.0, beta)

    filled[~mask] = beta * residual_values[~mask]
    stats["fill_beta"] = beta
    stats["filled_neurons"] = int((~mask).sum())
    return filled, stats


def _reshape_classifier_vector_to_tensor(vector, act_shape):
    arr = np.asarray(vector, dtype=np.float32).reshape(act_shape)
    return torch.from_numpy(arr).unsqueeze(0).to(device=IG.DEVICE, dtype=IG.DTYPE)


def _to_part_tuple(obj):
    if torch.is_tensor(obj):
        return (obj,)
    if isinstance(obj, (list, tuple)):
        tensors = tuple(part for part in obj if torch.is_tensor(part))
        if tensors:
            return tensors
    raise TypeError(f"Expected tensor/list/tuple, got {type(obj).__name__}")


def _flatten_part_arrays(parts):
    arrays = [part.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False) for part in parts]
    if not arrays:
        return np.empty(0, dtype=np.float64), []
    return np.concatenate(arrays, axis=0), [arr.size for arr in arrays]


def _split_flat_mask(flat_mask, part_sizes):
    masks = []
    start = 0
    for size in part_sizes:
        stop = start + int(size)
        masks.append(np.asarray(flat_mask[start:stop], dtype=bool))
        start = stop
    return masks


def _apply_sparse_mask_to_parts(parts, part_masks):
    sparse_parts = []
    for part, mask_np in zip(parts, part_masks):
        flat_part = part.reshape(-1)
        sparse_flat = torch.zeros_like(flat_part)
        if mask_np.any():
            mask_tensor = torch.from_numpy(mask_np).to(device=flat_part.device, dtype=torch.bool)
            sparse_flat[mask_tensor] = flat_part[mask_tensor]
        sparse_parts.append(sparse_flat.reshape(part.shape))
    if len(sparse_parts) == 1:
        return sparse_parts[0]
    return tuple(sparse_parts)


def _reshape_flat_vector_to_parts(flat_vector, template_parts, *, module_name):
    flat = np.asarray(flat_vector, dtype=np.float32)
    outputs = []
    start = 0
    for part in template_parts:
        size = int(part.numel())
        stop = start + size
        if stop > flat.size:
            raise ValueError(f"{module_name}: flat vector is shorter than required by template parts")
        piece = flat[start:stop].reshape(tuple(part.shape))
        outputs.append(torch.from_numpy(piece).to(device=part.device, dtype=part.dtype))
        start = stop
    if start != flat.size:
        raise ValueError(f"{module_name}: flat vector has unused tail elements")
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


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


def run_classifier_cheap_ig_pipeline(
    image_path,
    layer_name=IG.DEFAULT_LAYER_NAME,
    n_steps=128,
    baseline_mode="zero",
    baseline_rgb=None,
    baseline_blur_sigma=DEFAULT_BLUR_SIGMA,
    top_n=5,
    clear_every=8,
    verbose=False,
    show_total_plot=True,
    show_filter_plots=False,
    segment_start=0.0,
    segment_end=0.1,
    selection_mode="signed",
    selection_top_k=5000,
    fill_mode="zero",
    fill_rho=0.8,
    target_class_override=None,
):
    _validate_segment(n_steps, segment_start, segment_end)
    fill_mode = _normalize_fill_mode(fill_mode)

    x, image_np = IG.load_image(image_path)
    x0, baseline_info = build_image_baseline(
        x,
        image_np,
        mode=baseline_mode,
        baseline_rgb=baseline_rgb,
        blur_sigma=baseline_blur_sigma,
    )
    delta_x = (x - x0).contiguous()
    alphas = _alpha_grid(n_steps)
    segment_idx = _segment_index_mask(alphas, segment_start, segment_end)
    segment_index_set = {int(idx) for idx in np.asarray(segment_idx, dtype=np.int64).tolist()}

    hook = IG.LayerHook(IG.model, layer_name)
    try:
        def forward_with_layer(x_in):
            hook.clear()
            out = IG.model(x_in)
            act = IG.unwrap_tensor(hook.get())
            return out, act

        with torch.no_grad():
            out_x, act_x = forward_with_layer(x)
            _, logits = IG.split_classifier_output(out_x)
            if target_class_override is None:
                target_class = int(logits[0].argmax().item())
            else:
                target_class = int(target_class_override)
            target_name = IG.class_names[target_class]
            target_logit = float(logits[0, target_class].item())
            target_prob = float(torch.softmax(logits[0], dim=0)[target_class].item())

            out_x0, act_x0 = forward_with_layer(x0)
            _, logits_x0 = IG.split_classifier_output(out_x0)
            fx = float(logits[0, target_class].item())
            fx0 = float(logits_x0[0, target_class].item())

        act_shape = tuple(act_x.shape[1:])
        delta_y = (act_x - act_x0).detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False)
        fx_delta = fx - fx0

        sum_a_full = None
        sum_ab_segment = None
        segment_steps = 0

        def layer_activation(x_in):
            _, act = forward_with_layer(x_in)
            return act

        for step_idx, alpha in enumerate(alphas):
            x_alpha = (x0 + float(alpha) * delta_x).contiguous().detach().requires_grad_(True)

            out, act = forward_with_layer(x_alpha)
            _, logits = IG.split_classifier_output(out)
            score = logits[0, target_class]
            grad_y = torch.autograd.grad(score, act, retain_graph=False, create_graph=False)[0]
            grad_flat = grad_y.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False)

            if sum_a_full is None:
                sum_a_full = np.zeros_like(grad_flat, dtype=np.float64)
                sum_ab_segment = np.zeros_like(grad_flat, dtype=np.float64)

            sum_a_full += grad_flat

            if step_idx in segment_index_set:
                hook.clear()
                _, jvp_out = IG.jvp(layer_activation, (x_alpha,), (delta_x,))
                dir_flat = jvp_out.detach().cpu().reshape(-1).numpy().astype(np.float64, copy=False)
                sum_ab_segment += grad_flat * dir_flat
                segment_steps += 1
                del jvp_out

            del x_alpha, out, act, logits, score, grad_y
            hook.clear()

            if clear_every > 0 and ((step_idx + 1) % clear_every == 0):
                IG.clear_backend_cache()

        if sum_a_full is None or sum_ab_segment is None or segment_steps == 0:
            raise RuntimeError("cheap-ig failed to accumulate classifier attribution statistics")

        approx_naa_vector = (sum_a_full / len(alphas)) * delta_y
        exact_segment_vector = sum_ab_segment / segment_steps
        sparse_mask = _selection_mask(approx_naa_vector, selection_mode, selection_top_k)

        cheap_vector, fill_stats = _hybrid_fill_vector(
            exact_segment_vector,
            approx_naa_vector,
            sparse_mask,
            selection_mode=selection_mode,
            fill_mode=fill_mode,
            fill_rho=fill_rho,
        )
        cond_tensor = _reshape_classifier_vector_to_tensor(cheap_vector, act_shape)

        filter_scores = IG.reduce_filter_scores(cond_tensor)
        layer_score = cond_tensor.sum()
        topk = min(10, filter_scores.numel())
        top_vals, top_idx = torch.topk(filter_scores, k=topk)
        abs_error = abs(fx_delta - float(layer_score.item()))

        total_plot_title = "\n".join(
            [
                "Cheap-IG",
                f"layer={layer_name}",
                f"class={target_name}",
                f"segment=[{_format_float(segment_start)}, {_format_float(segment_end)}]",
                _selection_title_fragment(selection_mode, selection_top_k),
                _fill_title_fragment(fill_mode, fill_rho),
                baseline_title_fragment(
                    baseline_info["baseline_mode"],
                    baseline_rgb=baseline_info["baseline_rgb"],
                    blur_sigma=baseline_info["baseline_blur_sigma"],
                ),
                f"abs_error={abs_error:.6g}",
            ]
        )

        if verbose:
            print("image:", image_path)
            print("layer:", layer_name)
            print("target class:", target_class, target_name)
            print("target logit:", target_logit)
            print("target softmax prob:", target_prob)
            print("activation shape:", tuple(act_x.shape))
            print("segment steps:", segment_steps)
            print("selected neurons:", int(sparse_mask.sum()))
            print("fill mode:", fill_mode)
            print(
                "baseline:",
                baseline_title_fragment(
                    baseline_info["baseline_mode"],
                    baseline_rgb=baseline_info["baseline_rgb"],
                    blur_sigma=baseline_info["baseline_blur_sigma"],
                ),
            )
            print("fill beta:", float(fill_stats["fill_beta"]))
            print("F(x) - F(x0):", fx_delta)
            print("sum cheap-ig:", float(layer_score.item()))
            print("abs error:", abs_error)

        if show_total_plot:
            IG.plot_total_conductance_overlay(image_np, cond_tensor, title=total_plot_title)
        if show_filter_plots and top_n != 0:
            IG.plot_top_filter_overlays(
                image_np,
                cond_tensor,
                filter_scores,
                top_idx,
                top_n=top_n,
                show=True,
            )

        return {
            "image_path": image_path,
            "layer_name": layer_name,
            "target_class": target_class,
            "target_name": target_name,
            "target_logit": target_logit,
            "target_prob": target_prob,
            "activation_shape": tuple(act_x.shape),
            "cond_tensor": cond_tensor,
            "filter_scores": filter_scores,
            "layer_score": layer_score,
            "top_idx": top_idx,
            "top_vals": top_vals,
            "fx": fx,
            "fx0": fx0,
            "abs_error": abs_error,
            "baseline_mode": baseline_info["baseline_mode"],
            "baseline_rgb": baseline_info["baseline_rgb"],
            "baseline_blur_sigma": baseline_info["baseline_blur_sigma"],
            "image_np": image_np,
            "total_plot_title": total_plot_title,
            "segment_start": float(segment_start),
            "segment_end": float(segment_end),
            "segment_steps": int(segment_steps),
            "selection_mode": selection_mode,
            "selection_top_k": int(selection_top_k),
            "selected_neurons": int(sparse_mask.sum()),
            "fill_mode": fill_mode,
            "fill_rho": float(fill_rho),
            "fill_beta": float(fill_stats["fill_beta"]),
            "filled_neurons": int(fill_stats["filled_neurons"]),
            "approx_vector_sum": float(approx_naa_vector.sum()),
        }
    finally:
        hook.remove()
        IG.clear_backend_cache()


def run_detector_cheap_ig_pipeline(
    image_path,
    mode="fixed_roi_mean",
    layer_name=IG_det.DEFAULT_LAYER_NAME,
    n_steps=128,
    top_n=5,
    roi_top_k=-1,
    query_rank=None,
    query_head=None,
    bbox_iou_threshold=IG_det.BBOX_RANK_IOU_THRESHOLD,
    clear_every=8,
    verbose=False,
    show_total_plot=True,
    show_filter_plots=False,
    show_target_box=False,
    segment_start=0.0,
    segment_end=0.1,
    selection_mode="signed",
    selection_top_k=5000,
    fill_mode="zero",
    fill_rho=0.8,
):
    if mode not in {"fixed_query", "fixed_roi_lse", "fixed_roi_logmeanexp", "fixed_roi_mean"}:
        raise ValueError(f"Unsupported detector cheap-ig mode: {mode}")

    _validate_segment(n_steps, segment_start, segment_end)
    fill_mode = _normalize_fill_mode(fill_mode)

    if layer_name in IG_det.LAYER_GROUPS:
        resolved_layer_names = IG_det.LAYER_GROUPS[layer_name]
        resolved_layer_label = f"{layer_name} -> {resolved_layer_names}"
    else:
        resolved_layer_names = (layer_name,)
        resolved_layer_label = str(resolved_layer_names)

    x, image_np, _ = IG_det.load_image(image_path)
    x0 = IG_det.black_baseline_like(x)
    delta_x = (x - x0).contiguous()
    alphas = _alpha_grid(n_steps)
    segment_idx = _segment_index_mask(alphas, segment_start, segment_end)
    segment_index_set = {int(idx) for idx in np.asarray(segment_idx, dtype=np.int64).tolist()}
    bbox_iou_threshold = IG_det.BBOX_RANK_IOU_THRESHOLD if bbox_iou_threshold is None else float(bbox_iou_threshold)

    hook = IG_det.LayerHook(IG_det.model, resolved_layer_names)
    try:
        def forward_with_layer(x_in):
            hook.clear()
            out = IG_det.model(x_in)
            act = IG_det.unwrap_tensor(hook.get())
            return out, act

        with torch.no_grad():
            raw_x, act_x = forward_with_layer(x)
            parsed_x = IG_det.parse_detection_head(raw_x, num_classes=len(IG_det.class_names))
            target_spec = _resolve_detector_target_spec(
                raw_x,
                mode=mode,
                roi_top_k=roi_top_k,
                query_rank=query_rank,
                query_head=query_head,
                bbox_iou_threshold=bbox_iou_threshold,
            )
            fx = float(IG_det.detection_scalar_target(raw_x, target_spec, len(IG_det.class_names)).item())

            raw_x0, act_x0 = forward_with_layer(x0)
            fx0 = float(IG_det.detection_scalar_target(raw_x0, target_spec, len(IG_det.class_names)).item())

        fx_delta = fx - fx0
        target_class = int(target_spec["class_index"])
        target_name = IG_det.class_names[target_class]

        act_x_parts = _to_part_tuple(act_x)
        act_x0_parts = _to_part_tuple(act_x0)
        delta_y_parts = tuple((ax - ax0).detach() for ax, ax0 in zip(act_x_parts, act_x0_parts))
        grad_accum_parts = tuple(torch.zeros_like(part) for part in delta_y_parts)
        exact_segment_accum_parts = tuple(torch.zeros_like(part) for part in delta_y_parts)

        for step_idx, alpha in enumerate(alphas):
            x_alpha = (x0 + float(alpha) * delta_x).contiguous().detach().requires_grad_(True)

            raw_out, act = forward_with_layer(x_alpha)
            score = IG_det.detection_scalar_target(raw_out, target_spec, len(IG_det.class_names))
            act_parts = _to_part_tuple(act)
            grad_parts = torch.autograd.grad(
                score,
                act_parts,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            grad_parts = tuple(torch.zeros_like(a) if g is None else g for a, g in zip(act_parts, grad_parts))
            grad_accum_parts = tuple(acc + g.detach() for acc, g in zip(grad_accum_parts, grad_parts))

            if step_idx in segment_index_set:
                def layer_activation(x_in):
                    _, act_out = forward_with_layer(x_in)
                    return act_out

                hook.clear()
                _, jvp_out = IG_det.jvp(layer_activation, (x_alpha,), (delta_x,))
                dir_parts = _to_part_tuple(jvp_out)
                exact_parts = tuple(g.detach() * d.detach() for g, d in zip(grad_parts, dir_parts))
                exact_segment_accum_parts = tuple(
                    acc + part for acc, part in zip(exact_segment_accum_parts, exact_parts)
                )
                del jvp_out, dir_parts, exact_parts

            del x_alpha, raw_out, act, score, act_parts, grad_parts
            hook.clear()

            if clear_every > 0 and ((step_idx + 1) % clear_every == 0):
                IG_det.clear_backend_cache()

        approx_parts = tuple(dy * (acc / len(alphas)) for dy, acc in zip(delta_y_parts, grad_accum_parts))
        approx_vector, part_sizes = _flatten_part_arrays(approx_parts)
        flat_mask = _selection_mask(approx_vector, selection_mode, selection_top_k)
        part_masks = _split_flat_mask(flat_mask, part_sizes)

        segment_steps = int(len(segment_idx))
        if segment_steps == 0:
            raise RuntimeError("cheap-ig failed to accumulate detector segment attribution")

        exact_segment_parts = tuple(part / segment_steps for part in exact_segment_accum_parts)
        exact_segment_vector, _ = _flatten_part_arrays(exact_segment_parts)
        hybrid_vector, fill_stats = _hybrid_fill_vector(
            exact_segment_vector,
            approx_vector,
            flat_mask,
            selection_mode=selection_mode,
            fill_mode=fill_mode,
            fill_rho=fill_rho,
        )
        cond_tensor = _reshape_flat_vector_to_parts(
            hybrid_vector,
            exact_segment_parts,
            module_name="cheap_ig_detector",
        )

        filter_scores = IG_det.reduce_filter_scores(cond_tensor)
        layer_score = IG_det.sum_conductance_tensor(cond_tensor)
        topk = min(10, filter_scores.numel())
        top_vals, top_idx = torch.topk(filter_scores, k=topk)
        abs_error = abs(fx_delta - float(layer_score.item()))

        title_parts = [
            "Cheap-IG",
            f"layer={layer_name}",
            f"mode={mode}",
            f"class={target_name}",
            f"segment=[{_format_float(segment_start)}, {_format_float(segment_end)}]",
            _selection_title_fragment(selection_mode, selection_top_k),
            _fill_title_fragment(fill_mode, fill_rho),
        ]
        if mode != "fixed_query":
            title_parts.append(f"roi_top_k={int(roi_top_k)}")
        if query_rank is not None:
            title_parts.append(f"query_rank={int(query_rank)}")
        if query_head is not None:
            title_parts.append(f"query_head={query_head}")
        title_parts.append(f"abs_error={abs_error:.6g}")
        total_plot_title = "\n".join(title_parts)

        if mode == "fixed_query":
            target_box = target_spec["box_xywh"].detach().cpu().numpy()
            points_xy = None
            point_labels = None
            box_title = f"Fixed-query target box, class={target_name}"
        else:
            target_box = target_spec["fixed_box_xywh"].detach().cpu().numpy()
            roi_indices_cpu = target_spec["roi_indices"]
            roi_boxes = parsed_x["boxes"][0, roi_indices_cpu].detach().cpu().numpy()
            points_xy = [(float(b[0]), float(b[1])) for b in roi_boxes]
            point_labels = [str(i + 1) for i in range(len(points_xy))]
            box_title = f"Fixed ROI seed box, class={target_name}"

        if verbose:
            print("image:", image_path)
            print("layer request:", layer_name)
            print("resolved layer set:", resolved_layer_label)
            print("mode:", mode)
            print("target class:", target_class, target_name)
            print("segment steps:", segment_steps)
            print("selected neurons:", int(flat_mask.sum()))
            print("fill mode:", fill_mode)
            print("fill beta:", float(fill_stats["fill_beta"]))
            print("F(x) - F(x0):", fx_delta)
            print("sum cheap-ig:", float(layer_score.item()))
            print("abs error:", abs_error)

        if show_target_box:
            IG_det.draw_box_on_image(
                image_np,
                target_box,
                title=box_title,
                points_xy=points_xy,
                point_labels=point_labels,
                show=True,
            )
        if show_total_plot:
            IG_det.plot_total_conductance_overlay(
                image_np,
                cond_tensor,
                title=total_plot_title,
            )
        if show_filter_plots and top_n != 0:
            part_labels = list(resolved_layer_names) if len(resolved_layer_names) > 1 else None
            IG_det.plot_top_filter_overlays(
                image_np,
                cond_tensor,
                filter_scores,
                top_idx,
                top_n=top_n,
                part_labels=part_labels,
                show=True,
            )

        roi_query_count = int(target_spec["roi_mask"].sum().item()) if "roi_mask" in target_spec else None

        return {
            "image_path": image_path,
            "layer_name": layer_name,
            "resolved_layer_names": tuple(resolved_layer_names),
            "target_class": target_class,
            "target_name": target_name,
            "target_spec": target_spec,
            "activation_shape": tuple(tuple(part.shape) for part in act_x_parts),
            "cond_tensor": cond_tensor,
            "filter_scores": filter_scores,
            "layer_score": layer_score,
            "top_idx": top_idx,
            "top_vals": top_vals,
            "fx": fx,
            "fx0": fx0,
            "abs_error": abs_error,
            "image_np": image_np,
            "total_plot_title": total_plot_title,
            "segment_start": float(segment_start),
            "segment_end": float(segment_end),
            "segment_steps": segment_steps,
            "selection_mode": selection_mode,
            "selection_top_k": int(selection_top_k),
            "selected_neurons": int(flat_mask.sum()),
            "fill_mode": fill_mode,
            "fill_rho": float(fill_rho),
            "fill_beta": float(fill_stats["fill_beta"]),
            "filled_neurons": int(fill_stats["filled_neurons"]),
            "mode": mode,
            "roi_top_k": int(roi_top_k),
            "query_rank": query_rank,
            "query_head": query_head,
            "bbox_iou_threshold": float(bbox_iou_threshold),
            "roi_query_count": roi_query_count,
        }
    finally:
        hook.remove()
        IG_det.clear_backend_cache()
