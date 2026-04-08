# %% [markdown]
# # Реализация кода из статьи [Improving Adversarial Transferability via Neuron Attribution-Based Attacks](https://arxiv.org/pdf/2204.00008)

# %% [markdown]
# ### Импорты и базовая конфигурация
# Подключаем библиотеки, включаем градиенты и определяем устройство для вычислений.
# 

# %%
import math
import gc

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from torch.func import jvp
from torchvision.ops import nms as tv_nms

torch.set_grad_enabled(True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
DTYPE = torch.float32

# %% [markdown]
# ### Загрузка модели классификации
# Инициализируем YOLO-классификатор, переводим модель на нужное устройство и проверяем классы.
# 

# %%
# Загрузка детектора
yolo = YOLO("yolo11s.pt")
model = yolo.model.to(DEVICE).eval()
class_names = yolo.names

# %% [markdown]
# ### Подготовка входа и служебные утилиты
# Определяем размер изображения, функции загрузки/препроцессинга и очистки кэша backend.
# 

# %%
IMG_SIZE = 640
DEFAULT_LAYER_NAME = "prehead_all"
BBOX_RANK_IOU_THRESHOLD = 0.7
HEAD_NAME_TO_HW = {
    "80x80": (80, 80),
    "40x40": (40, 40),
    "20x20": (20, 20),
}
LAYER_GROUPS = {
    # Backbone stages
    "backbone_p2": ("model.2",),
    "backbone_p3": ("model.4",),
    "backbone_p4": ("model.6",),
    "backbone_p5": ("model.10",),

    # Neck fused layers (после слияния ветвей)
    "neck_td_p4_concat": ("model.12",),
    "neck_td_p4_out": ("model.13",),
    "neck_td_p3_concat": ("model.15",),
    "neck_td_p3_out": ("model.16",),
    "neck_bu_p4_concat": ("model.18",),
    "neck_bu_p4_out": ("model.19",),
    "neck_bu_p5_concat": ("model.21",),
    "neck_bu_p5_out": ("model.22",),

    # Pre-head single-scale outputs
    "prehead_p3": ("model.16",),
    "prehead_p4": ("model.19",),
    "prehead_p5": ("model.22",),

    # Главный кандидат: полный multi-scale слой перед Detect
    "prehead_all": ("model.16", "model.19", "model.22"),
}

HEAD_NAME_TO_HW = {
    "80x80": (80, 80),
    "40x40": (40, 40),
    "20x20": (20, 20),
}

HEAD_NAME_TO_PREHEAD_LAYER = {
    "80x80": "model.16",
    "40x40": "model.19",
    "20x20": "model.22",
}

def load_image(path, img_size=IMG_SIZE, pad_value=114 / 255.0):
    """Load an image, apply detector-style letterbox resizing, and return tensor, RGB image, and resize metadata."""
    img = Image.open(path).convert("RGB")
    img_np = np.asarray(img).astype(np.float32) / 255.0
    h, w = img_np.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Некорректный размер изображения: {(h, w)}")

    scale = img_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = np.asarray(
        Image.fromarray((img_np * 255).astype(np.uint8)).resize((new_w, new_h), Image.Resampling.BILINEAR)
    ).astype(np.float32) / 255.0

    canvas = np.full((img_size, img_size, 3), pad_value, dtype=np.float32)
    top = (img_size - new_h) // 2
    left = (img_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    x = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).to(DEVICE, DTYPE)
    meta = {
        "orig_hw": (h, w),
        "resized_hw": (new_h, new_w),
        "pad_top_left": (top, left),
        "scale": scale,
    }
    return x, canvas, meta


def black_baseline_like(x):
    """Create a zero baseline tensor with the same shape and device as the input."""
    return torch.zeros_like(x)

def list_named_modules(model):
    """Return all non-empty named submodules of the given model."""
    return [(name, module) for name, module in model.named_modules() if name != ""]

def clear_backend_cache():
    """Trigger Python garbage collection and clear CUDA/MPS caches when available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

# %% [markdown]
# ### Хуки, парсинг выходов и визуализация
# Добавляем вспомогательные функции для получения активаций слоя, агрегации и построения карт attribution.
# 

# %%
# Вспомогательные утилиты для layer hook, output parsing и plotting

def unwrap_tensor(output):
    """Extract the first tensor object from a model output container."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Не удалось извлечь tensor activation из output типа {type(output).__name__}")

def _collect_tensors(obj):
    tensors = []

    def _collect(x):
        if torch.is_tensor(x):
            tensors.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                _collect(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                _collect(v)

    _collect(obj)
    return tensors


def unpack_detector_output(raw_output):
    """
    Явно разбирает output Ultralytics YOLO detector.
    Поддерживаем:
    - tensor y
    - (y, preds)
    """
    if torch.is_tensor(raw_output):
        return raw_output, None

    if isinstance(raw_output, (list, tuple)):
        if len(raw_output) == 0:
            raise RuntimeError("Пустой output у detector.")

        y = raw_output[0] if torch.is_tensor(raw_output[0]) else None
        preds = raw_output[1] if len(raw_output) > 1 else None

        if y is None:
            first_tensor = None
            for item in raw_output:
                if torch.is_tensor(item):
                    first_tensor = item
                    break
            if first_tensor is None:
                raise RuntimeError("Не удалось найти основной detector output tensor в tuple/list output.")
            y = first_tensor

        return y, preds

    raise RuntimeError(f"Неожиданный тип detector output: {type(raw_output).__name__}")


def normalize_detection_y(y, num_classes):
    """Приводит основной detector output к форме [B, Q, D]."""
    shape = tuple(y.shape)
    d_min = 4 + num_classes

    if y.ndim == 3:
        b, a, c = y.shape

        if a == d_min and c != d_min:
            return y.transpose(1, 2)
        if c == d_min and a != d_min:
            return y
        if a >= d_min and c < d_min:
            return y.transpose(1, 2)
        if c >= d_min and a < d_min:
            return y
        if a >= d_min and c >= d_min:
            if a <= c:
                return y.transpose(1, 2)
            return y

        raise RuntimeError(f"Не удалось интерпретировать 3D detector output shape={shape}")

    if y.ndim == 4:
        if y.shape[1] >= d_min:
            b, d, h, w = y.shape
            return y.permute(0, 2, 3, 1).reshape(b, h * w, d)

        if y.shape[-1] >= d_min:
            b, h, w, d = y.shape
            return y.reshape(b, h * w, d)

        raise RuntimeError(f"Не удалось интерпретировать 4D detector output shape={shape}")

    raise RuntimeError(f"Ожидался 3D или 4D detector output, получено shape={shape}")


def parse_detection_head(raw_output, num_classes):
    """Возвращает нормализованное dense detector prediction и индексацию queries по трём YOLO-scale головам."""
    y, preds = unpack_detector_output(raw_output)
    y_shape = tuple(y.shape)
    pred = normalize_detection_y(y, num_classes=num_classes)
    pred_shape = tuple(pred.shape)

    if pred.shape[-1] < 4 + num_classes:
        raise RuntimeError(
            f"Последняя размерность слишком мала: ожидалось >= {4 + num_classes}, получено {pred.shape[-1]}"
        )

    boxes = pred[..., :4]
    cls_logits = pred[..., -num_classes:]

    q_total = pred.shape[1]
    expected_total = sum(h * w for h, w in HEAD_NAME_TO_HW.values())
    if q_total != expected_total:
        raise RuntimeError(
            f"Ожидалось {expected_total} queries для трёх голов YOLO, получено {q_total}. "
            f"Проверь соответствие detector output и HEAD_NAME_TO_HW."
        )

    head_slices = {}
    start = 0
    for head_name in ("80x80", "40x40", "20x20"):
        h, w = HEAD_NAME_TO_HW[head_name]
        count = h * w
        head_slices[head_name] = slice(start, start + count)
        start += count

    return {
        "y": y,
        "y_shape": y_shape,
        "pred": pred,
        "pred_shape": pred_shape,
        "boxes": boxes,
        "cls_logits": cls_logits,
        "preds": preds,
        "head_slices": head_slices,
    }


def layer_shape_repr(obj):
    if torch.is_tensor(obj):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [tuple(x.shape) if torch.is_tensor(x) else type(x).__name__ for x in obj]
    return type(obj).__name__


def sum_conductance_tensor(cond_tensor):
    if torch.is_tensor(cond_tensor):
        return cond_tensor.sum()
    if isinstance(cond_tensor, (list, tuple)):
        return sum(part.sum() for part in cond_tensor)
    raise TypeError(f"cond_tensor должен быть tensor/list/tuple, получено {type(cond_tensor).__name__}")

class LayerHook:
    def __init__(self, model, layer_name):
        """Initialize forward hooks for one layer or a tuple/list of layers and allocate activation storage."""
        self.layer_name = layer_name
        self.layer_store = {}
        modules = dict(model.named_modules())
        self.handles = []

        if isinstance(layer_name, str):
            layer_names = (layer_name,)
        elif isinstance(layer_name, (list, tuple)):
            layer_names = tuple(layer_name)
        else:
            raise TypeError(f"layer_name должен быть str/list/tuple, получено {type(layer_name).__name__}")

        self.layer_names = layer_names
        for name in layer_names:
            if name not in modules:
                raise KeyError(f"Слой '{name}' не найден в model.named_modules()")
            self.handles.append(modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def _hook(module, inp, out):
            self.layer_store[name] = out
        return _hook

    def clear(self):
        self.layer_store.clear()

    def get(self):
        if len(self.layer_names) == 1:
            return self.layer_store[self.layer_names[0]]
        return tuple(self.layer_store[name] for name in self.layer_names)

    def remove(self):
        for handle in self.handles:
            handle.remove()

def reduce_filter_scores(cond_tensor):
    """Aggregate attribution over non-filter axes to produce one score per filter, supporting both tensor and tuple/list outputs."""
    if torch.is_tensor(cond_tensor):
        parts = (cond_tensor,)
    elif isinstance(cond_tensor, (list, tuple)):
        parts = tuple(cond_tensor)
    else:
        raise TypeError(f"cond_tensor должен быть tensor/list/tuple, получено {type(cond_tensor).__name__}")

    scores = []
    for part in parts:
        if part.ndim < 2:
            raise ValueError(
                f"Ожидался tensor с batch-осью и хотя бы одной feature-осью, получено shape={tuple(part.shape)}"
            )
        per_sample = part[0]
        if per_sample.ndim == 1:
            scores.append(per_sample)
        else:
            reduce_dims = tuple(range(1, per_sample.ndim))
            scores.append(per_sample.sum(dim=reduce_dims))

    return torch.cat(scores, dim=0)

def _normalize_map(arr):
    """Normalize a 2D map by its maximum absolute value for stable visualization."""
    arr = arr.astype(np.float32)
    max_abs = np.max(np.abs(arr))
    if max_abs > 0:
        arr = arr / max_abs
    return arr

def _resize_map_nearest(arr, out_hw):
    # Без сглаживания: nearest-neighbor upsampling
    """Resize a 2D map to target height and width using nearest-neighbor sampling."""
    h_out, w_out = out_hw
    h_in, w_in = arr.shape

    row_idx = np.floor(np.arange(h_out) * (h_in / h_out)).astype(int)
    col_idx = np.floor(np.arange(w_out) * (w_in / w_out)).astype(int)

    row_idx = np.clip(row_idx, 0, h_in - 1)
    col_idx = np.clip(col_idx, 0, w_in - 1)

    return arr[row_idx][:, col_idx]

def build_total_attribution_overlay_figure(image_np, cond_tensor, title=None):
    if torch.is_tensor(cond_tensor):
        parts = [cond_tensor]
    elif isinstance(cond_tensor, (list, tuple)):
        parts = list(cond_tensor)
    else:
        raise TypeError(f"cond_tensor должен быть tensor/list/tuple, получено {type(cond_tensor).__name__}")

    spatial_maps = []
    for part in parts:
        cond_np = part[0].detach().cpu().numpy()
        if cond_np.ndim >= 3:
            total_map = cond_np.sum(axis=0)
            spatial_maps.append(_resize_map_nearest(_normalize_map(total_map), image_np.shape[:2]))

    if not spatial_maps:
        return None

    merged = np.mean(np.stack(spatial_maps, axis=0), axis=0)
    merged = _normalize_map(merged)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np, interpolation="nearest")
    heat = ax.imshow(merged, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title or "Total attribution overlay")

    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized attribution", rotation=90)
    return fig


def plot_total_conductance_overlay(image_np, cond_tensor, title=None):
    fig = build_total_attribution_overlay_figure(image_np, cond_tensor, title=title)
    if fig is None:
        return None
    plt.show()
    return fig


def plot_top_filter_overlays(image_np, cond_tensor, filter_scores, top_idx, top_n=5, part_labels=None, show=True):
    if torch.is_tensor(cond_tensor):
        parts = [cond_tensor]
    elif isinstance(cond_tensor, (list, tuple)):
        parts = list(cond_tensor)
    else:
        raise TypeError(f"cond_tensor должен быть tensor/list/tuple, получено {type(cond_tensor).__name__}")

    spatial_parts = []
    offset = 0
    for part_idx, part in enumerate(parts):
        cond_np = part[0].detach().cpu().numpy()
        channels = cond_np.shape[0]
        label = part_labels[part_idx] if part_labels is not None and part_idx < len(part_labels) else f"part {part_idx + 1}"
        if cond_np.ndim >= 3:
            spatial_parts.append((offset, offset + channels, cond_np, label))
        offset += channels

    if not spatial_parts:
        return []

    if top_n == -1:
        selected_idx = top_idx.tolist()
    else:
        top_n = min(top_n, len(top_idx))
        selected_idx = top_idx[:top_n].tolist()
    figures = []

    for rank, idx in enumerate(selected_idx, start=1):
        chosen_map = None
        source_label = None
        local_idx = None
        grid_hw = None
        for start, end, cond_np, label in spatial_parts:
            if start <= idx < end:
                local_idx = idx - start
                chosen_map = cond_np[local_idx]
                source_label = label
                grid_hw = tuple(chosen_map.shape)
                break

        if chosen_map is None:
            continue

        fmap = _normalize_map(chosen_map)
        fmap = _resize_map_nearest(fmap, image_np.shape[:2])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np, interpolation="nearest")
        ax.imshow(fmap, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f"Filter {idx} [{source_label}, local {local_idx}, grid {grid_hw[0]}x{grid_hw[1]}] attribution = {float(filter_scores[idx]):+.6f} (rank {rank})"
        )
        figures.append(fig)

        if show:
            plt.show()

    return figures

# %%
def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def build_ranked_bbox_candidates(parsed, num_classes, query_head=None, iou_threshold=BBOX_RANK_IOU_THRESHOLD):
    cls_logits = parsed["cls_logits"][0].detach().cpu()
    boxes_xywh = parsed["boxes"][0].detach().cpu()

    if query_head is None:
        query_indices = torch.arange(cls_logits.shape[0], dtype=torch.long)
    else:
        head_slice = parsed["head_slices"][query_head]
        query_indices = torch.arange(head_slice.start, head_slice.stop, dtype=torch.long)

    cls_subset = cls_logits[query_indices]
    best_scores, best_classes = torch.max(cls_subset, dim=1)
    boxes_subset_xywh = boxes_xywh[query_indices]

    boxes_subset_xyxy = torch.empty_like(boxes_subset_xywh)
    boxes_subset_xyxy[:, 0] = boxes_subset_xywh[:, 0] - boxes_subset_xywh[:, 2] / 2.0
    boxes_subset_xyxy[:, 1] = boxes_subset_xywh[:, 1] - boxes_subset_xywh[:, 3] / 2.0
    boxes_subset_xyxy[:, 2] = boxes_subset_xywh[:, 0] + boxes_subset_xywh[:, 2] / 2.0
    boxes_subset_xyxy[:, 3] = boxes_subset_xywh[:, 1] + boxes_subset_xywh[:, 3] / 2.0

    keep_global = []
    unique_classes = torch.unique(best_classes)
    for class_idx in unique_classes.tolist():
        class_mask = best_classes == class_idx
        class_positions = torch.nonzero(class_mask, as_tuple=False).flatten()
        class_boxes = boxes_subset_xyxy[class_mask]
        class_scores = best_scores[class_mask]

        kept_local = tv_nms(class_boxes, class_scores, float(iou_threshold))
        keep_global.append(class_positions[kept_local])

    if keep_global:
        keep_global = torch.cat(keep_global, dim=0)
    else:
        keep_global = torch.empty(0, dtype=torch.long)

    kept_scores = best_scores[keep_global]
    sorted_order = torch.argsort(kept_scores, descending=True)
    keep_global = keep_global[sorted_order]

    selected = []
    for pos in keep_global.tolist():
        q_idx = int(query_indices[pos].item())
        q_local = int(pos)
        class_idx = int(best_classes[pos].item())
        score = float(best_scores[pos].item())
        box_xywh = boxes_subset_xywh[pos]
        box_xyxy = boxes_subset_xyxy[pos]

        selected.append({
            "query_index": q_idx,
            "query_index_local": q_local,
            "query_head": query_head,
            "class_index": class_idx,
            "score": score,
            "box_xywh": box_xywh.clone(),
            "box_xyxy": box_xyxy.clone(),
        })

    return selected


def make_roi_mask_from_fixed_box(boxes_xywh, fixed_box_xywh, center_only=True):
    boxes = boxes_xywh.detach().cpu().numpy()
    fixed_xyxy = xywh_to_xyxy(np.asarray(fixed_box_xywh))

    cx = boxes[:, 0]
    cy = boxes[:, 1]
    mask = (
        (cx >= fixed_xyxy[0]) &
        (cx <= fixed_xyxy[2]) &
        (cy >= fixed_xyxy[1]) &
        (cy <= fixed_xyxy[3])
    )
    return torch.from_numpy(mask)


def pick_fixed_roi_target(
    raw_output,
    num_classes,
    min_queries=1,
    roi_mode="fixed_roi_mean",
    roi_top_k=-1,
    query_rank=None,
    query_head=None,
    bbox_iou_threshold=BBOX_RANK_IOU_THRESHOLD,
):
    if roi_mode not in {"fixed_roi_lse", "fixed_roi_logmeanexp", "fixed_roi_mean"}:
        raise ValueError(f"Неизвестный roi_mode: {roi_mode}")
    if roi_top_k == 0 or roi_top_k < -1:
        raise ValueError(f"roi_top_k должен быть -1 или положительным целым, получено {roi_top_k}")
    if query_rank is not None and query_rank < 1:
        raise ValueError(f"query_rank должен быть >= 1, получено {query_rank}")
    if query_head is not None and query_head not in HEAD_NAME_TO_HW:
        raise ValueError(f"Неизвестный query_head={query_head!r}. Ожидается one of {tuple(HEAD_NAME_TO_HW.keys())} или None")

    parsed = parse_detection_head(raw_output, num_classes=num_classes)
    cls_logits = parsed["cls_logits"][0]
    boxes = parsed["boxes"][0]

    if query_rank is None:
        if query_head is None:
            candidate_logits = cls_logits
            query_offset = 0
        else:
            head_slice = parsed["head_slices"][query_head]
            candidate_logits = cls_logits[head_slice]
            query_offset = head_slice.start

        flat_scores = candidate_logits.reshape(-1)
        flat_idx_local = int(torch.argmax(flat_scores).item())
        best_q_local = flat_idx_local // num_classes
        class_idx = flat_idx_local % num_classes
        best_q = query_offset + best_q_local
        chosen_box_xyxy = torch.tensor(xywh_to_xyxy(boxes[best_q].detach().cpu().numpy()))
        bbox_candidates_count = None
        bbox_selection_mode = "raw top-1 query-class pair"
    else:
        ranked_bboxes = build_ranked_bbox_candidates(
            parsed,
            num_classes=num_classes,
            query_head=query_head,
            iou_threshold=bbox_iou_threshold,
        )

        if len(ranked_bboxes) < query_rank:
            raise ValueError(
                f"query_rank={query_rank} больше числа доступных bbox-кандидатов ({len(ranked_bboxes)}) "
                f"для query_head={query_head}"
            )

        chosen = ranked_bboxes[query_rank - 1]
        best_q = int(chosen["query_index"])
        best_q_local = int(chosen["query_index_local"])
        class_idx = int(chosen["class_index"])
        chosen_box_xyxy = chosen["box_xyxy"]
        bbox_candidates_count = len(ranked_bboxes)
        bbox_selection_mode = "best-class-per-query + class-aware NMS"

    fixed_box = boxes[best_q].detach().cpu().numpy()
    roi_mask = make_roi_mask_from_fixed_box(boxes, fixed_box, center_only=True)

    if query_head is not None:
        head_slice = parsed["head_slices"][query_head]
        head_mask = torch.zeros_like(roi_mask, dtype=torch.bool)
        head_mask[head_slice] = True
        roi_mask = roi_mask & head_mask

    if roi_mask.sum().item() < min_queries:
        roi_mask[best_q] = True

    roi_indices = torch.nonzero(roi_mask, as_tuple=False).flatten()
    roi_scores = cls_logits[roi_indices, class_idx]

    if roi_top_k != -1 and roi_indices.numel() > roi_top_k:
        top_local = torch.topk(roi_scores, k=roi_top_k).indices
        top_local = top_local.to(roi_indices.device)
        roi_indices = roi_indices[top_local]
        new_roi_mask = torch.zeros_like(roi_mask, dtype=torch.bool)
        new_roi_mask[roi_indices] = True
        roi_mask = new_roi_mask
        roi_scores = cls_logits[roi_indices, class_idx]

    selected = roi_scores
    if roi_mode == "fixed_roi_lse":
        pooled_score = float(torch.logsumexp(selected, dim=0).item())
    elif roi_mode == "fixed_roi_logmeanexp":
        pooled_score = float((torch.logsumexp(selected, dim=0) - math.log(float(selected.numel()))).item())
    else:
        pooled_score = float(selected.mean().item())

    return {
        "mode": roi_mode,
        "class_index": class_idx,
        "seed_query_index": best_q,
        "seed_query_index_local": best_q_local,
        "query_head": query_head,
        "query_rank": 1 if query_rank is None else query_rank,
        "fixed_box_xywh": torch.tensor(fixed_box),
        "fixed_box_xyxy": chosen_box_xyxy,
        "roi_mask": roi_mask.clone(),
        "roi_indices": roi_indices.detach().cpu(),
        "roi_top_k": roi_top_k,
        "score": pooled_score,
        "bbox_iou_threshold": bbox_iou_threshold,
        "bbox_candidates_count": bbox_candidates_count,
        "bbox_selection_mode": bbox_selection_mode,
        "y_shape": parsed["y_shape"],
        "pred_shape": parsed["pred_shape"],
    }


def detection_scalar_target(raw_output, target_spec, num_classes):
    parsed = parse_detection_head(raw_output, num_classes=num_classes)
    cls_logits = parsed["cls_logits"][0]
    c = target_spec["class_index"]
    roi_mask = target_spec["roi_mask"].to(cls_logits.device)
    selected = cls_logits[roi_mask, c]

    if target_spec["mode"] == "fixed_roi_lse":
        return torch.logsumexp(selected, dim=0)
    if target_spec["mode"] == "fixed_roi_logmeanexp":
        return torch.logsumexp(selected, dim=0) - math.log(float(selected.numel()))
    if target_spec["mode"] == "fixed_roi_mean":
        return selected.mean()

    raise ValueError(f"Неизвестный mode: {target_spec['mode']}")


def build_box_figure(image_np, box_xywh, title=None, points_xy=None, point_labels=None):
    img = image_np.copy()
    h, w = img.shape[:2]

    x, y, bw, bh = [float(v) for v in box_xywh]
    x1 = int(round(x - bw / 2))
    y1 = int(round(y - bh / 2))
    x2 = int(round(x + bw / 2))
    y2 = int(round(y + bh / 2))

    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, interpolation="nearest")
    ax.add_patch(
        plt.Rectangle((x1, y1), max(1, x2 - x1), max(1, y2 - y1), fill=False, linewidth=2, edgecolor="lime")
    )

    if points_xy is not None and len(points_xy) > 0:
        cmap = plt.cm.get_cmap("tab10", max(1, len(points_xy)))
        for i, (px, py) in enumerate(points_xy):
            color = cmap(i)
            ax.scatter([px], [py], s=50, color=color, edgecolors="white", linewidths=1.0, zorder=5)
            if point_labels is not None and i < len(point_labels):
                ax.text(
                    px + 4,
                    py - 4,
                    str(point_labels[i]),
                    color=color,
                    fontsize=9,
                    weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, ec="none"),
                    zorder=6,
                )

    ax.axis("off")
    ax.set_title(title or "Target box")
    return fig


def draw_box_on_image(image_np, box_xywh, title=None, points_xy=None, point_labels=None, show=True):
    fig = build_box_figure(
        image_np,
        box_xywh,
        title=title,
        points_xy=points_xy,
        point_labels=point_labels,
    )
    if show:
        plt.show()
    return fig

# %% [markdown]
# #### Особенности реализации кодом
# В статье далее предлагается не считать точную neuron conductance по формуле выше, а перейти к приближению, которое существенно дешевле вычислительно.
# 
# Для каждого нейрона $y_j$ вводится
# $$
# IA(y_j)
# :=
# \frac{1}{n}\sum_{m=1}^{n}
# \frac{\partial F}{\partial y_j}(y(x_m)),
# \qquad
# x_m = x' + \frac{m}{n}(x-x').
# $$
# 
# Это и есть **Integrated Attention**: средний градиент выхода модели по данному нейрону вдоль пути от baseline к входу.
# 
# Далее статья делает ключевое приближение:
# $$
# A_{y_j}
# \approx
# \Delta y_j \cdot IA(y_j),
# \qquad
# \Delta y_j = y_j - y'_j,
# $$
# где $y'_j$ — активация того же нейрона на baseline-входе $x'$. Тогда для всего слоя
# $$
# A_y
# \approx
# \sum_{y_j\in y} \Delta y_j IA(y_j)
# =
# (y-y')\odot IA(y),
# $$
# где $\odot$ обозначает поэлементное произведение.
# 
# Именно это приближение и используется в коде:
# 1. один раз считаем baseline-активацию $y'$,
# 2. усредняем градиенты $\frac{\partial F}{\partial y}(y(x_m))$ по нескольким точкам пути,
# 3. умножаем полученное $IA(y)$ на $(y-y')$.
# 
# Это уже не точная conductance-формула из *How Important Is a Neuron?*, а приближённая neuron attribution-схема из NAA.

# %%
def compute_detector_naa_attribution(
    model,
    hook,
    x,
    x0,
    target_spec,
    num_classes,
    n_steps=30,
    clear_every=8,
):
    """Estimate detector layer attribution with the NAA approximation: A_y ≈ (y - y0) * IA."""
    x = x.contiguous()
    x0 = x0.contiguous()
    delta_x = (x - x0).contiguous()

    def forward_with_layer(x_in):
        hook.clear()
        out = model(x_in)
        act = hook.get()
        return out, act

    def to_part_tuple(obj):
        if torch.is_tensor(obj):
            return (obj,)
        if isinstance(obj, (list, tuple)):
            tensors = tuple(x for x in obj if torch.is_tensor(x))
            if not tensors:
                raise TypeError("Не удалось извлечь tensor parts из composite activation")
            return tensors
        raise TypeError(f"Ожидался tensor/list/tuple, получено {type(obj).__name__}")

    with torch.no_grad():
        _, act_x = forward_with_layer(x)
        _, act_x0 = forward_with_layer(x0)
        act_x_parts = to_part_tuple(act_x)
        act_x0_parts = to_part_tuple(act_x0)
        delta_y_parts = tuple((ax - ax0).detach() for ax, ax0 in zip(act_x_parts, act_x0_parts))

    ia_accum_parts = tuple(torch.zeros_like(part) for part in delta_y_parts)

    for m in range(1, n_steps + 1):
        alpha = m / n_steps
        x_alpha = (x0 + alpha * delta_x).contiguous().detach().requires_grad_(True)

        raw_out, act = forward_with_layer(x_alpha)
        score = detection_scalar_target(raw_out, target_spec=target_spec, num_classes=num_classes)

        act_parts = to_part_tuple(act)
        grad_parts = torch.autograd.grad(
            score, act_parts, retain_graph=False, create_graph=False, allow_unused=True
        )
        grad_parts = tuple(torch.zeros_like(a) if g is None else g for a, g in zip(act_parts, grad_parts))
        ia_accum_parts = tuple(acc + g.detach() for acc, g in zip(ia_accum_parts, grad_parts))

        del x_alpha, raw_out, act, score, act_parts, grad_parts
        hook.clear()

        if m % clear_every == 0:
            clear_backend_cache()

    ia_parts = tuple(acc / n_steps for acc in ia_accum_parts)
    attr_parts = tuple(dy * ia for dy, ia in zip(delta_y_parts, ia_parts))
    clear_backend_cache()

    if len(attr_parts) == 1:
        return attr_parts[0], ia_parts[0], delta_y_parts[0]
    return attr_parts, ia_parts, delta_y_parts

# %% [markdown]
# ### Единый pipeline эксперимента
# Собираем полный процесс: предсказание, attribution, ранжирование фильтров и построение графиков.
# 

# %%
def run_attribution_pipeline(
    image_path,
    mode="fixed_roi_mean",
    layer_name=DEFAULT_LAYER_NAME,
    n_steps=30,
    top_n=5,
    roi_top_k=-1,
    query_rank=None,
    query_head=None,
    bbox_iou_threshold=BBOX_RANK_IOU_THRESHOLD,
    clear_every=8,
    verbose=False,
    show_total_plot=True,
    show_filter_plots=False,
    show_target_box=False,
):
    if mode not in {"fixed_roi_lse", "fixed_roi_logmeanexp", "fixed_roi_mean"}:
        raise ValueError(f"Неизвестный mode: {mode}")

    if layer_name in LAYER_GROUPS:
        resolved_layer_names = LAYER_GROUPS[layer_name]
        resolved_layer_label = f"{layer_name} -> {resolved_layer_names}"
    else:
        resolved_layer_names = (layer_name,)
        resolved_layer_label = str(resolved_layer_names)

    x, img_np, meta = load_image(image_path)
    x0 = black_baseline_like(x)

    hook = LayerHook(model, resolved_layer_names)

    try:
        def forward_with_layer(x_in):
            hook.clear()
            out = model(x_in)
            act = hook.get()
            return out, act

        with torch.no_grad():
            raw_out, act = forward_with_layer(x)
            parsed_x = parse_detection_head(raw_out, num_classes=len(class_names))
            target_spec = pick_fixed_roi_target(
                raw_out,
                num_classes=len(class_names),
                roi_mode=mode,
                roi_top_k=roi_top_k,
                query_rank=query_rank,
                query_head=query_head,
                bbox_iou_threshold=bbox_iou_threshold,
            )

            c = target_spec["class_index"]
            target_class_name = class_names[c]
            seed_q = target_spec["seed_query_index"]
            target_box = target_spec["fixed_box_xywh"].numpy()
            roi_count = int(target_spec["roi_mask"].sum().item())
            target_score = target_spec["score"]
            roi_indices_cpu = target_spec["roi_indices"]
            roi_boxes = parsed_x["boxes"][0, roi_indices_cpu].detach().cpu().numpy()
            points_xy = [(float(b[0]), float(b[1])) for b in roi_boxes]
            point_labels = [str(i + 1) for i in range(len(points_xy))]

        cond_tensor, ia_tensor, delta_y = compute_detector_naa_attribution(
            model=model,
            hook=hook,
            x=x,
            x0=x0,
            target_spec=target_spec,
            num_classes=len(class_names),
            n_steps=n_steps,
            clear_every=clear_every,
        )

        filter_scores = reduce_filter_scores(cond_tensor)
        layer_score = sum_conductance_tensor(cond_tensor)

        filter_layer_labels = []
        if torch.is_tensor(cond_tensor):
            part_tensors = (cond_tensor,)
        else:
            part_tensors = tuple(cond_tensor)

        for part_idx, part in enumerate(part_tensors):
            channels = int(part.shape[1])
            layer_label = resolved_layer_names[part_idx] if part_idx < len(resolved_layer_names) else f"part_{part_idx}"
            filter_layer_labels.extend([layer_label] * channels)

        topk = filter_scores.numel() if top_n == -1 else min(10, filter_scores.numel())
        top_pos_vals, top_pos_idx = torch.topk(filter_scores, k=topk)
        top_neg_vals, top_neg_idx = torch.topk(-filter_scores, k=topk)
        top_neg_vals = -top_neg_vals

        with torch.no_grad():
            raw_x, _ = forward_with_layer(x)
            raw_x0, _ = forward_with_layer(x0)
            fx = float(detection_scalar_target(raw_x, target_spec, len(class_names)).item())
            fx0 = float(detection_scalar_target(raw_x0, target_spec, len(class_names)).item())
            abs_error = abs((fx - fx0) - float(layer_score.item()))

        box_title = f"Fixed ROI seed box, class={target_class_name}"
        overlay_title = (
            f"Total attribution, layer={layer_name}, mode={mode}, class={target_class_name}, abs_error={abs_error:.6g}"
        )

        if verbose:
            print("image:", image_path)
            print("layer request:", layer_name)
            print("resolved layer set:", resolved_layer_label)
            print("target class:", c, target_class_name)
            print("seed query index:", seed_q)
            print("ROI query count:", roi_count)
            print("roi_top_k:", roi_top_k)
            print("target pooled score at x:", target_score)
            print("layer activation shape:", layer_shape_repr(act))
            print("cond tensor shape:", layer_shape_repr(cond_tensor))
            print("filter_scores shape:", tuple(filter_scores.shape))

            print("\ntop positive filters by attribution:")
            preview_n = filter_scores.numel() if top_n == -1 else min(10, len(top_pos_idx))
            for rank, idx in enumerate(top_pos_idx[:preview_n].tolist(), start=1):
                val = float(filter_scores[idx].item())
                print(f"{rank:2d}. filter {idx:4d} [{filter_layer_labels[idx]}]: {val:+.6f}")

            print("\ntop negative filters by attribution:")
            preview_n_neg = filter_scores.numel() if top_n == -1 else min(10, len(top_neg_idx))
            for rank, idx in enumerate(top_neg_idx[:preview_n_neg].tolist(), start=1):
                val = float(filter_scores[idx].item())
                print(f"{rank:2d}. filter {idx:4d} [{filter_layer_labels[idx]}]: {val:+.6f}")

            print("\nlayer attribution sum:", float(layer_score.item()))
            print("F(x)            =", fx)
            print("F(x0)           =", fx0)
            print("F(x) - F(x0)    =", fx - fx0)
            print("sum attribution =", float(layer_score.item()))
            print("abs error       =", abs_error)

        if show_target_box:
            draw_box_on_image(
                img_np,
                target_box,
                title=box_title,
                points_xy=points_xy,
                point_labels=point_labels,
                show=True,
            )
        if show_total_plot:
            plot_total_conductance_overlay(
                img_np,
                cond_tensor,
                title=overlay_title,
            )
        if show_filter_plots and top_n != 0:
            plot_top_filter_overlays(
                img_np,
                cond_tensor,
                filter_scores,
                top_pos_idx,
                top_n=top_n,
                part_labels=resolved_layer_names,
                show=True,
            )

        return {
            "mode": mode,
            "image_path": image_path,
            "layer_name": layer_name,
            "resolved_layer_names": resolved_layer_names,
            "target_spec": target_spec,
            "cond_tensor": cond_tensor,
            "ia_tensor": ia_tensor,
            "delta_y": delta_y,
            "filter_scores": filter_scores,
            "filter_layer_labels": filter_layer_labels,
            "layer_score": layer_score,
            "top_pos_idx": top_pos_idx,
            "top_pos_vals": top_pos_vals,
            "top_neg_idx": top_neg_idx,
            "top_neg_vals": top_neg_vals,
            "fx": fx,
            "fx0": fx0,
            "abs_error": abs_error,
            "image_np": img_np,
            "total_plot_title": overlay_title,
            "box_title": box_title,
            "target_box": target_box,
            "points_xy": points_xy,
            "point_labels": point_labels,
        }

    finally:
        hook.remove()
        clear_backend_cache()

# %% [markdown]
# ## Эксперименты
# 
if __name__ == "__main__":

    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_attribution_pipeline(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=64,
        top_n=5,
        roi_top_k=1,
    )

    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_attribution_pipeline(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=64,
        top_n=5,
        roi_top_k=5,
    )

    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_attribution_pipeline(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="model.22",
        n_steps=64,
        top_n=5,
        roi_top_k=5,
    )

    # %%
    result_prehead_all = run_attribution_pipeline(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=64,
        top_n=5,
        roi_top_k=5,
        query_rank=2
    )

    # %%
    result_prehead_all = run_attribution_pipeline(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=128,
        top_n=5,
        roi_top_k=5,
        query_head='20x20'
    )

    # %%
    result_prehead_all = run_attribution_pipeline(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=128,
        top_n=5,
        roi_top_k=5,
        query_head='40x40'
    )

    # %%
    result_prehead_all = run_attribution_pipeline(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=128,
        top_n=5,
        roi_top_k=5,
        query_head='80x80'
    )

    # %% [markdown]
    # ## Проверка сходимости ошибки

    # %%
    step_grid = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    abs_errors = []
    results_by_steps = {}

    for n_steps in step_grid:
        print(f"\n=== n_steps = {n_steps} ===")
        result = run_attribution_pipeline(
            image_path="data/person.png",
            mode="fixed_roi_mean",
            layer_name="model.22",
            n_steps=n_steps,
            top_n=0,
            roi_top_k=5,
        )
        abs_errors.append(result["abs_error"])
        results_by_steps[n_steps] = result


    # %%
    plt.figure(figsize=(8, 5))
    plt.plot(step_grid, abs_errors, marker="o")
    plt.xscale("log", base=2)
    plt.xticks(step_grid, step_grid, rotation=45)
    plt.xlabel("Number of integration steps")
    plt.ylabel("Absolute error")
    plt.title("Conductance absolute error vs integration steps on log scale\nimage=jess.jpg, layer=model.5")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    print("\nSummary:")
    for n_steps, err in zip(step_grid, abs_errors):
        print(f"n_steps={n_steps:5d} | abs_error={err:.10f}")
