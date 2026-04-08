# %% [markdown]
# # Реализация кода из статьи Google [How Important Is a Neuron?](https://arxiv.org/abs/1805.12233), перенесенная на детектор

# %%
import gc
import math
import warnings

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

# %%
# Загрузка детектора
yolo = YOLO("yolo11s.pt")
model = yolo.model.to(DEVICE).eval()
class_names = yolo.names
num_classes_global = len(class_names)

# %%
IMG_SIZE = 640
DEFAULT_LAYER_NAME = "prehead_all"
BBOX_RANK_IOU_THRESHOLD = 0.7

# Осмысленные "полные" слои / наборы слоёв для YOLO11s.
# Идея:
# - брать завершённые backbone stage,
# - fused-слои после Concat,
# - либо полный multi-scale pre-head срез перед Detect.
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

def load_image(path, img_size=IMG_SIZE, pad_value=0.0):
    img = Image.open(path).convert("RGB")
    img_np = np.asarray(img).astype(np.uint8)

    h, w = img_np.shape[:2]
    if h == 0 or w == 0:
        raise ValueError(f"Некорректный размер изображения: {(h, w)}")

    scale = img_size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = np.asarray(
        Image.fromarray(img_np).resize((new_w, new_h), Image.Resampling.BICUBIC)
    ).astype(np.float32) / 255.0

    canvas = np.full((img_size, img_size, 3), pad_value, dtype=np.float32)
    top = (img_size - new_h) // 2
    left = (img_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized

    x = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).to(DEVICE, DTYPE)
    meta = {
        "orig_hw": (h, w),
        "resized_hw": (new_h, new_w),
        "pad_top": top,
        "pad_left": left,
        "scale": scale,
        "img_size": img_size,
    }
    return x, canvas, meta

def black_baseline_like(x):
    return torch.zeros_like(x)

def clear_backend_cache():
    gc.collect()
    if DEVICE.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if DEVICE.type == "mps" and hasattr(torch, "mps") and torch.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

# %%
def print_main_yolo_layers(model, sample_input, max_depth=2):
    """
    Печатает основные слои YOLO и shape их выходов на одном forward-pass.
    - depth=1: model.0, model.1, ...
    - depth=2: model.15.cv1, model.15.cv2, ...
    """
    module_outputs = {}
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            module_outputs[name] = out
        return hook

    try:
        for name, module in model.named_modules():
            if name == "":
                continue
            if not name.startswith("model."):
                continue

            depth = name.count(".")
            if depth <= max_depth:
                handles.append(module.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = model(sample_input)

        print("Main YOLO layers:\n")
        for name, module in model.named_modules():
            if name == "":
                continue
            if not name.startswith("model."):
                continue

            depth = name.count(".")
            if depth <= max_depth:
                out = module_outputs.get(name, None)
                if torch.is_tensor(out):
                    shape_str = str(tuple(out.shape))
                elif isinstance(out, (list, tuple)):
                    tensor_shapes = [tuple(x.shape) for x in out if torch.is_tensor(x)]
                    shape_str = str(tensor_shapes) if tensor_shapes else f"<{type(out).__name__}>"
                elif out is None:
                    shape_str = "<no output captured>"
                else:
                    shape_str = f"<{type(out).__name__}>"

                print(f"{name:25s} {type(module).__name__:12s} output={shape_str}")
    finally:
        for h in handles:
            h.remove()

# Вспомогательные функции для hooks и plotting

class LayerHook:
    def __init__(self, model, layer_name_or_names):
        self.layer_names = self._normalize_layer_names(layer_name_or_names)
        self.layer_store = {}
        self.handles = []
        modules = dict(model.named_modules())

        for layer_name in self.layer_names:
            if layer_name not in modules:
                raise KeyError(f"Слой '{layer_name}' не найден в model.named_modules()")
            self.handles.append(modules[layer_name].register_forward_hook(self._make_hook(layer_name)))

    @staticmethod
    def _normalize_layer_names(layer_name_or_names):
        if isinstance(layer_name_or_names, str):
            return (layer_name_or_names,)
        if isinstance(layer_name_or_names, (list, tuple)):
            return tuple(layer_name_or_names)
        raise TypeError(
            f"layer_name_or_names должен быть str/list/tuple, получено {type(layer_name_or_names).__name__}"
        )

    def _make_hook(self, layer_name):
        def hook(module, inp, out):
            self.layer_store[layer_name] = out
        return hook

    def clear(self):
        self.layer_store.clear()

    def get(self):
        if len(self.layer_names) == 1:
            return self.layer_store[self.layer_names[0]]
        return tuple(self.layer_store[name] for name in self.layer_names)

    def remove(self):
        for h in self.handles:
            h.remove()


def unwrap_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        tensor_items = []
        for item in output:
            if torch.is_tensor(item):
                tensor_items.append(item)
            elif isinstance(item, (list, tuple)):
                for sub_item in item:
                    if torch.is_tensor(sub_item):
                        tensor_items.append(sub_item)
        if len(tensor_items) == 1:
            return tensor_items[0]
        if len(tensor_items) > 1:
            return tuple(tensor_items)
    raise TypeError(f"Не удалось извлечь tensor activation из output типа {type(output).__name__}")


def reduce_filter_scores(cond_tensor):
    if torch.is_tensor(cond_tensor):
        tensors = (cond_tensor,)
    elif isinstance(cond_tensor, (list, tuple)):
        tensors = tuple(cond_tensor)
    else:
        raise TypeError(f"cond_tensor должен быть tensor/list/tuple, получено {type(cond_tensor).__name__}")

    reduced_parts = []
    for part in tensors:
        if part.ndim < 2:
            raise ValueError(
                f"Ожидался tensor с batch-осью и хотя бы одной feature-осью, получено shape={tuple(part.shape)}"
            )

        per_sample = part[0]
        if per_sample.ndim == 1:
            reduced_parts.append(per_sample)
        else:
            reduce_dims = tuple(range(1, per_sample.ndim))
            reduced_parts.append(per_sample.sum(dim=reduce_dims))

    if len(reduced_parts) == 1:
        return reduced_parts[0]
    return torch.cat(reduced_parts, dim=0)


def layer_shape_repr(act):
    if torch.is_tensor(act):
        return tuple(act.shape)
    if isinstance(act, (list, tuple)):
        return tuple(tuple(x.shape) for x in act if torch.is_tensor(x))
    return f"<{type(act).__name__}>"


def sum_conductance_tensor(cond_tensor):
    if torch.is_tensor(cond_tensor):
        return cond_tensor.sum()
    if isinstance(cond_tensor, (list, tuple)):
        total = None
        for part in cond_tensor:
            part_sum = part.sum()
            total = part_sum if total is None else (total + part_sum)
        return total
    raise TypeError(f"cond_tensor должен быть tensor/list/tuple, получено {type(cond_tensor).__name__}")


def _normalize_map(arr):
    arr = arr.astype(np.float32)
    max_abs = np.max(np.abs(arr))
    if max_abs > 0:
        arr = arr / max_abs
    return arr


def _resize_map_nearest(arr, out_hw):
    h_out, w_out = out_hw
    h_in, w_in = arr.shape

    row_idx = np.floor(np.arange(h_out) * (h_in / h_out)).astype(int)
    col_idx = np.floor(np.arange(w_out) * (w_in / w_out)).astype(int)

    row_idx = np.clip(row_idx, 0, h_in - 1)
    col_idx = np.clip(col_idx, 0, w_in - 1)
    return arr[row_idx][:, col_idx]


def build_total_conductance_overlay_figure(image_np, cond_tensor, title=None):
    if torch.is_tensor(cond_tensor):
        parts = [cond_tensor]
    elif isinstance(cond_tensor, (list, tuple)):
        parts = list(cond_tensor)
    else:
        raise TypeError(f"cond_tensor должен быть tensor/list/tuple, получено {type(cond_tensor).__name__}")

    spatial_parts = []
    for part in parts:
        cond_np = part[0].detach().cpu().numpy()
        if cond_np.ndim >= 3:
            spatial_parts.append(cond_np.sum(axis=0))

    if not spatial_parts:
        return None

    total_map = spatial_parts[0]
    for extra_map in spatial_parts[1:]:
        if extra_map.shape != total_map.shape:
            extra_map = _resize_map_nearest(extra_map, total_map.shape)
        total_map = total_map + extra_map

    total_map = _normalize_map(total_map)
    total_map = _resize_map_nearest(total_map, image_np.shape[:2])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np, interpolation="nearest")
    heat = ax.imshow(total_map, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title or "Total conductance overlay")

    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized signed conductance", rotation=90)
    return fig


def plot_total_conductance_overlay(image_np, cond_tensor, title=None):
    fig = build_total_conductance_overlay_figure(image_np, cond_tensor, title=title)
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
            f"Filter {idx} [{source_label}, local {local_idx}, grid {grid_hw[0]}x{grid_hw[1]}] conductance = {float(filter_scores[idx]):+.6f} (rank {rank})"
        )
        figures.append(fig)

        if show:
            plt.show()

    return figures

# %%
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

    Ожидаемые варианты:
    - tensor y
    - (y, preds)

    где:
    - y: основной dense prediction tensor головы детектора
    - preds: дополнительные сырые/промежуточные данные головы

    Мы больше НЕ выбираем "самый большой tensor", потому что это легко приводит
    к выбору feature map вместо detector output.
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
    """
    Приводит основной detector output к форме [B, Q, D].

    Поддерживаем:
    - [B, Q, D]
    - [B, D, Q]
    - [B, D, H, W] -> [B, H*W, D]
    - [B, H, W, D] -> [B, H*W, D]

    Важно: для inference output YOLO вида (1, 84, 8400)
    это формат [B, D, Q], а НЕ [B, Q, D].
    Раньше код ошибочно оставлял его как есть, и тогда:
    - первые 4 QUERY интерпретировались как box coords,
    - последние 80 QUERY интерпретировались как class logits.
    Именно это ломало target selection и bbox.
    """
    shape = tuple(y.shape)
    d_min = 4 + num_classes

    if y.ndim == 3:
        b, a, c = y.shape

        # Случай [B, D, Q], типичный для YOLO inference output, например (1, 84, 8400)
        # Здесь axis=1 -- channel dim (4 + num_classes), axis=2 -- число queries.
        if a == d_min and c != d_min:
            return y.transpose(1, 2)

        # Случай [B, Q, D]
        if c == d_min and a != d_min:
            return y

        # Более общий fallback:
        # если только одна из осей {1,2} похожа на channel dim, используем её как D.
        if a >= d_min and c < d_min:
            return y.transpose(1, 2)
        if c >= d_min and a < d_min:
            return y

        # Если обе оси >= d_min, выбираем меньшую как D, потому что
        # channel dim обычно 4+num_classes, а query dim обычно сильно больше.
        if a >= d_min and c >= d_min:
            if a <= c:
                return y.transpose(1, 2)
            return y

        raise RuntimeError(f"Не удалось интерпретировать 3D detector output shape={shape}")

    if y.ndim == 4:
        # [B, D, H, W]
        if y.shape[1] >= d_min:
            b, d, h, w = y.shape
            return y.permute(0, 2, 3, 1).reshape(b, h * w, d)

        # [B, H, W, D]
        if y.shape[-1] >= d_min:
            b, h, w, d = y.shape
            return y.reshape(b, h * w, d)

        raise RuntimeError(f"Не удалось интерпретировать 4D detector output shape={shape}")

    raise RuntimeError(f"Ожидался 3D или 4D detector output, получено shape={shape}")


def parse_detection_head(raw_output, num_classes):
    """
    Возвращает нормализованное dense detector prediction из основного выхода головы
    и дополнительно индексирует queries по трём YOLO-scale головам.
    """
    y, preds = unpack_detector_output(raw_output)
    y_shape = tuple(y.shape)
    pred = normalize_detection_y(y, num_classes=num_classes)
    pred_shape = tuple(pred.shape)

    if pred.shape[-1] < 4 + num_classes:
        raise RuntimeError(
            f"Последняя размерность слишком мала: ожидалось >= {4 + num_classes}, получено {pred.shape[-1]}"
        )

    boxes = pred[..., :4]                  # [B, Q, 4]
    cls_logits = pred[..., -num_classes:]  # [B, Q, C]

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

# %%
# Выбор target на конечном изображении.
#
# Мы фиксируем target по x = исходной картинке,
# а потом используем тот же target на всех x(α).

def pick_fixed_query_target(
    raw_output,
    num_classes,
    query_rank=None,
    query_head=None,
    bbox_iou_threshold=BBOX_RANK_IOU_THRESHOLD,
):
    if query_rank is not None and query_rank < 1:
        raise ValueError(f"query_rank должен быть >= 1, получено {query_rank}")
    if query_head is not None and query_head not in HEAD_NAME_TO_HW:
        raise ValueError(f"Неизвестный query_head={query_head!r}. Ожидается one of {tuple(HEAD_NAME_TO_HW.keys())} или None")

        parsed = parse_detection_head(raw_output, num_classes=num_classes)

    if query_rank is None:
        cls_logits = parsed["cls_logits"][0]  # [Q, C]

        if query_head is None:
            candidate_logits = cls_logits
            query_offset = 0
        else:
            head_slice = parsed["head_slices"][query_head]
            candidate_logits = cls_logits[head_slice]
            query_offset = head_slice.start

        flat_scores = candidate_logits.reshape(-1)
        flat_idx_local = int(torch.argmax(flat_scores).item())
        q_idx_local = flat_idx_local // num_classes
        class_idx = flat_idx_local % num_classes
        q_idx = query_offset + q_idx_local
        score = float(cls_logits[q_idx, class_idx].item())
        box_xywh = parsed["boxes"][0, q_idx].detach().cpu()
        box_xyxy = torch.tensor(xywh_to_xyxy(box_xywh.numpy()))

        chosen = {
            "query_index": q_idx,
            "query_index_local": q_idx_local,
            "class_index": class_idx,
            "score": score,
            "box_xywh": box_xywh,
            "box_xyxy": box_xyxy,
        }
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
        bbox_candidates_count = len(ranked_bboxes)
        bbox_selection_mode = "best-class-per-query + class-aware NMS"

    return {
        "mode": "fixed_query",
        "query_index": chosen["query_index"],
        "query_index_local": chosen["query_index_local"],
        "query_head": query_head,
        "query_rank": 1 if query_rank is None else query_rank,
        "class_index": chosen["class_index"],
        "score": chosen["score"],
        "box_xywh": chosen["box_xywh"],
        "box_xyxy": chosen["box_xyxy"],
        "bbox_iou_threshold": bbox_iou_threshold,
        "bbox_candidates_count": bbox_candidates_count,
        "bbox_selection_mode": bbox_selection_mode,
        "y_shape": parsed["y_shape"],
        "pred_shape": parsed["pred_shape"],
    }


def xywh_to_xyxy(xywh):
    x, y, w, h = xywh
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def boxes_xywh_to_xyxy(boxes_xywh):
    boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)
    xyxy = np.empty_like(boxes_xywh, dtype=np.float32)
    xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    return xyxy


def build_ranked_bbox_candidates(parsed, num_classes, query_head=None, iou_threshold=BBOX_RANK_IOU_THRESHOLD):
    """
    Строит ранжированный список bbox-кандидатов.

    ВАЖНО: ранжирование идёт по bbox (query), а не по всем query-class pair.
    Для каждого query берём его лучший класс и лучший score, после чего делаем
    class-aware NMS. Это резко быстрее прежней схемы, где перебирались все Q*C
    комбинации и NMS выполнялся greedily в Python.
    """
    cls_logits = parsed["cls_logits"][0].detach().cpu()   # [Q, C]
    boxes_xywh = parsed["boxes"][0].detach().cpu()        # [Q, 4]

    if query_head is None:
        query_indices = torch.arange(cls_logits.shape[0], dtype=torch.long)
    else:
        head_slice = parsed["head_slices"][query_head]
        query_indices = torch.arange(head_slice.start, head_slice.stop, dtype=torch.long)

    cls_subset = cls_logits[query_indices]                # [Qh, C]
    best_scores, best_classes = torch.max(cls_subset, dim=1)
    boxes_subset_xywh = boxes_xywh[query_indices]         # [Qh, 4]

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

    if center_only:
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        mask = (
            (cx >= fixed_xyxy[0]) &
            (cx <= fixed_xyxy[2]) &
            (cy >= fixed_xyxy[1]) &
            (cy <= fixed_xyxy[3])
        )
    else:
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
    roi_mode="fixed_roi_lse",
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

# %%
# Scalar target F(x) для разных режимов
#
# fixed_query:
#   F(x) = s[q*, c]
#
# fixed_roi_lse:
#   F(x) = logsumexp({s[q, c] : q in ROI*})
#
# fixed_roi_logmeanexp:
#   F(x) = logsumexp({s[q, c] : q in ROI*}) - log(|ROI*|)
#
# fixed_roi_mean:
#   F(x) = mean({s[q, c] : q in ROI*})
#
# Важно: logmeanexp и mean обычно дают более стабильную baseline-нормировку,
# чем чистый logsumexp по большому ROI.

def detection_scalar_target(raw_output, target_spec, num_classes):
    parsed = parse_detection_head(raw_output, num_classes=num_classes)
    cls_logits = parsed["cls_logits"][0]  # [Q, C]

    if target_spec["mode"] == "fixed_query":
        q = target_spec["query_index"]
        c = target_spec["class_index"]
        return cls_logits[q, c]

    if target_spec["mode"] in {"fixed_roi_lse", "fixed_roi_logmeanexp", "fixed_roi_mean"}:
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

# %%
def compute_detector_conductance(
    model,
    hook,
    x,
    x0,
    target_spec,
    num_classes,
    n_steps=64,
    fd_eps=1e-3,
    clear_every=8,
    warn_on_fallback=False,
):
    x = x.contiguous()
    x0 = x0.contiguous()
    delta_x = (x - x0).contiguous()

    def forward_with_layer(x_in):
        hook.clear()
        out = model(x_in)
        act = unwrap_tensor(hook.get())
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

    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device, dtype=x.dtype)
    step = 1.0 / n_steps

    cond_accum_parts = None
    used_fallback = False

    for k in range(n_steps):
        alpha = (alphas[k] + alphas[k + 1]) / 2.0
        x_alpha = (x0 + alpha * delta_x).contiguous().detach().requires_grad_(True)

        raw_out, act = forward_with_layer(x_alpha)
        score = detection_scalar_target(raw_out, target_spec=target_spec, num_classes=num_classes)

        act_parts = to_part_tuple(act)
        grad_parts = torch.autograd.grad(
            score, act_parts, retain_graph=False, create_graph=False, allow_unused=True
        )
        grad_parts = tuple(torch.zeros_like(a) if g is None else g for a, g in zip(act_parts, grad_parts))

        try:
            def act_only(inp):
                inp = inp.contiguous()
                _, a = forward_with_layer(inp)
                return to_part_tuple(a)

            _, act_tangent_parts = jvp(act_only, (x_alpha,), (delta_x,))
            act_tangent_parts = to_part_tuple(act_tangent_parts)

        except RuntimeError as e:
            if "view size is not compatible with input tensor's size and stride" not in str(e):
                raise

            used_fallback = True

            alpha_plus = min(float(alpha.item()) + fd_eps, 1.0)
            alpha_minus = max(float(alpha.item()) - fd_eps, 0.0)
            denom = alpha_plus - alpha_minus

            if denom == 0.0:
                raise RuntimeError("Не удалось построить finite-difference fallback для dy/dalpha.")

            with torch.no_grad():
                _, act_plus = forward_with_layer((x0 + alpha_plus * delta_x).contiguous())
                _, act_minus = forward_with_layer((x0 + alpha_minus * delta_x).contiguous())

            act_plus_parts = to_part_tuple(act_plus)
            act_minus_parts = to_part_tuple(act_minus)
            act_tangent_parts = tuple((ap - am) / denom for ap, am in zip(act_plus_parts, act_minus_parts))
            del act_plus, act_minus, act_plus_parts, act_minus_parts

        integrand_parts = tuple(g * t for g, t in zip(grad_parts, act_tangent_parts))

        if cond_accum_parts is None:
            cond_accum_parts = tuple(part.detach() for part in integrand_parts)
        else:
            cond_accum_parts = tuple(acc + part.detach() for acc, part in zip(cond_accum_parts, integrand_parts))

        del x_alpha, raw_out, act, score, act_parts, grad_parts, act_tangent_parts, integrand_parts
        hook.clear()

        if (k + 1) % clear_every == 0:
            clear_backend_cache()

    cond_parts = tuple(part * step for part in cond_accum_parts)
    clear_backend_cache()

    if used_fallback and warn_on_fallback:
        warnings.warn(
            "jvp на текущем backend не сработал, для dy/dalpha использована конечная разность.",
            stacklevel=2,
        )

    if len(cond_parts) == 1:
        return cond_parts[0]
    return cond_parts

# %%
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


def run_detector_conductance(
    image_path,
    mode="fixed_query",
    layer_name=DEFAULT_LAYER_NAME,
    n_steps=64,
    top_n=5,
    roi_top_k=-1,
    query_rank=None,
    query_head=None,
    bbox_iou_threshold=BBOX_RANK_IOU_THRESHOLD,
    fd_eps=1e-3,
    clear_every=8,
    verbose=False,
    show_total_plot=True,
    show_filter_plots=False,
    show_target_box=False,
    warn_on_fallback=False,
):
    if mode not in {"fixed_query", "fixed_roi_lse", "fixed_roi_logmeanexp", "fixed_roi_mean"}:
        raise ValueError(f"Неизвестный mode: {mode}")
    if query_rank is not None and query_rank < 1:
        raise ValueError(f"query_rank должен быть >= 1, получено {query_rank}")
    if query_head is not None and query_head not in HEAD_NAME_TO_HW:
        raise ValueError(f"Неизвестный query_head={query_head!r}. Ожидается one of {tuple(HEAD_NAME_TO_HW.keys())} или None")
    if bbox_iou_threshold < 0.0 or bbox_iou_threshold > 1.0:
        raise ValueError(f"bbox_iou_threshold должен быть в [0, 1], получено {bbox_iou_threshold}")

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
            act = unwrap_tensor(hook.get())
            return out, act

        with torch.no_grad():
            raw_out, act = forward_with_layer(x)
            parsed_x = parse_detection_head(raw_out, num_classes=len(class_names))
            points_xy = None
            point_labels = None

            if mode == "fixed_query":
                target_spec = pick_fixed_query_target(
                    raw_out,
                    num_classes=len(class_names),
                    query_rank=query_rank,
                    query_head=query_head,
                    bbox_iou_threshold=bbox_iou_threshold,
                )

                q = target_spec["query_index"]
                c = target_spec["class_index"]
                target_class_name = class_names[c]
                target_score = target_spec["score"]
                target_box = target_spec["box_xywh"].numpy()

                extra_lines = [
                    ("mode", "fixed_query"),
                    ("query_head", target_spec["query_head"]),
                    ("query_rank", target_spec["query_rank"]),
                    ("bbox_iou_threshold", target_spec["bbox_iou_threshold"]),
                    ("bbox candidates count", target_spec["bbox_candidates_count"]),
                    ("bbox selection mode", target_spec["bbox_selection_mode"]),
                    ("target query index", q),
                    ("target query local index", target_spec["query_index_local"]),
                    ("target class", f"{c} {target_class_name}"),
                    ("target raw score at x", target_score),
                    ("detector y shape", target_spec["y_shape"]),
                    ("normalized pred shape", target_spec["pred_shape"]),
                ]

            else:
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

                extra_lines = [
                    ("mode", mode),
                    ("query_head", target_spec["query_head"]),
                    ("query_rank", target_spec["query_rank"]),
                    ("bbox_iou_threshold", target_spec["bbox_iou_threshold"]),
                    ("bbox candidates count", target_spec["bbox_candidates_count"]),
                    ("bbox selection mode", target_spec["bbox_selection_mode"]),
                    ("seed query index", seed_q),
                    ("seed query local index", target_spec["seed_query_index_local"]),
                    ("target class", f"{c} {target_class_name}"),
                    ("ROI query count", roi_count),
                    ("roi_top_k", target_spec["roi_top_k"]),
                    ("target pooled score at x", target_score),
                    ("detector y shape", target_spec["y_shape"]),
                    ("normalized pred shape", target_spec["pred_shape"]),
                ]

        cond_tensor = compute_detector_conductance(
            model=model,
            hook=hook,
            x=x,
            x0=x0,
            target_spec=target_spec,
            num_classes=len(class_names),
            n_steps=n_steps,
            fd_eps=fd_eps,
            clear_every=clear_every,
            warn_on_fallback=warn_on_fallback,
        )

        filter_scores = reduce_filter_scores(cond_tensor)
        layer_score = sum_conductance_tensor(cond_tensor)
        filter_layer_labels = []
        if torch.is_tensor(cond_tensor):
            part_tensors = (cond_tensor,)
        elif isinstance(cond_tensor, (list, tuple)):
            part_tensors = tuple(cond_tensor)
        else:
            raise TypeError(f"cond_tensor должен быть tensor/list/tuple, получено {type(cond_tensor).__name__}")

        for part_idx, part in enumerate(part_tensors):
            channels = int(part.shape[1])
            if part_idx < len(resolved_layer_names):
                layer_label = resolved_layer_names[part_idx]
            else:
                layer_label = f"part_{part_idx}"
            filter_layer_labels.extend([layer_label] * channels)

        if top_n == -1:
            topk = filter_scores.numel()
        else:
            topk = min(10, filter_scores.numel())

        top_pos_vals, top_pos_idx = torch.topk(filter_scores, k=topk)
        top_neg_vals, top_neg_idx = torch.topk(-filter_scores, k=topk)
        top_neg_vals = -top_neg_vals

        with torch.no_grad():
            raw_x, _ = forward_with_layer(x)
            raw_x0, _ = forward_with_layer(x0)
            fx = float(detection_scalar_target(raw_x, target_spec, len(class_names)).item())
            fx0 = float(detection_scalar_target(raw_x0, target_spec, len(class_names)).item())
            abs_error = abs((fx - fx0) - float(layer_score.item()))

        box_title = (
            f"Fixed-query target box, class={target_class_name}"
            if mode == "fixed_query"
            else f"Fixed ROI seed box, class={target_class_name}"
        )
        overlay_title = (
            f"Total conductance, layer={layer_name}, mode={mode}, class={target_class_name}, abs_error={abs_error:.6g}"
        )

        if verbose:
            print("image:", image_path)
            print("layer request:", layer_name)
            print("resolved layer set:", resolved_layer_label)
            for key, value in extra_lines:
                print(f"{key}: {value}")
            print("layer activation shape:", layer_shape_repr(act))
            print("cond tensor shape:", layer_shape_repr(cond_tensor))
            print("filter_scores shape:", tuple(filter_scores.shape))

            print("\ntop positive filters by conductance:")
            preview_n = filter_scores.numel() if top_n == -1 else min(10, len(top_pos_idx))
            for rank, idx in enumerate(top_pos_idx[:preview_n].tolist(), start=1):
                val = float(filter_scores[idx].item())
                print(f"{rank:2d}. filter {idx:4d} [{filter_layer_labels[idx]}]: {val:+.6f}")

            print("\ntop negative filters by conductance:")
            preview_n_neg = filter_scores.numel() if top_n == -1 else min(10, len(top_neg_idx))
            for rank, idx in enumerate(top_neg_idx[:preview_n_neg].tolist(), start=1):
                val = float(filter_scores[idx].item())
                print(f"{rank:2d}. filter {idx:4d} [{filter_layer_labels[idx]}]: {val:+.6f}")

            print("\nlayer conductance sum:", float(layer_score.item()))
            print("F(x)            =", fx)
            print("F(x0)           =", fx0)
            print("F(x) - F(x0)    =", fx - fx0)
            print("sum conductance =", float(layer_score.item()))
            print("abs error       =", abs_error)

            with torch.no_grad():
                cls_logits_x = parsed_x["cls_logits"][0]
                topk_flat = torch.topk(cls_logits_x.reshape(-1), k=min(10, cls_logits_x.numel()))
                print("\nTop raw detector queries by class score:")
                for rank, flat_idx in enumerate(topk_flat.indices.tolist(), start=1):
                    q_idx = flat_idx // len(class_names)
                    c_idx = flat_idx % len(class_names)
                    score = float(cls_logits_x[q_idx, c_idx].item())
                    box = parsed_x["boxes"][0, q_idx].detach().cpu().numpy()
                    print(
                        f"{rank:2d}. query={q_idx:4d} class={c_idx:3d} ({class_names[c_idx]}) score={score:+.6f} box_xywh={box}"
                    )

                if mode != "fixed_query":
                    roi_indices = target_spec["roi_indices"]
                    print("\nSelected ROI queries:")
                    for rank, q_idx in enumerate(roi_indices.tolist(), start=1):
                        score = float(cls_logits_x[q_idx, target_spec["class_index"]].item())
                        box = parsed_x["boxes"][0, q_idx].detach().cpu().numpy()
                        print(
                            f"{rank:2d}. query={q_idx:4d} class={target_spec['class_index']:3d} ({target_class_name}) score={score:+.6f} box_xywh={box}"
                        )

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
            plot_total_conductance_overlay(img_np, cond_tensor, title=overlay_title)
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

if __name__ == "__main__":
    # %%
    result_prehead_all = run_detector_conductance(
        image_path="data/jess.jpg",
        mode="fixed_roi_mean",
        layer_name="model.22",
        n_steps=128,
        top_n=5,
        roi_top_k=5,
    )

    # %%
    result_prehead_all = run_detector_conductance(
        image_path="data/fox.jpg",
        mode="fixed_roi_mean",
        layer_name="model.22",
        n_steps=128,
        top_n=5,
        roi_top_k=5,
    )

    # %%
    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=128,
        top_n=5,
        roi_top_k=1,
    )

    # %%
    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=1024,
        top_n=5,
        roi_top_k=5,
    )

    # %%
    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="model.22",
        n_steps=1024,
        top_n=5,
        roi_top_k=5,
    )

    # %%
    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="model.22",
        n_steps=1024,
        top_n=5,
        roi_top_k=1,
    )

    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=128,
        top_n=5,
        roi_top_k=5, # changed
    )

    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=128,
        top_n=5,
        roi_top_k=5,
        query_rank=2
    )

    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=128,
        top_n=5,
        roi_top_k=5,
        query_head='20x20'
    )

    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
        image_path="data/person.png",
        mode="fixed_roi_mean",
        layer_name="prehead_all",
        n_steps=128,
        top_n=5,
        roi_top_k=5,
        query_head='40x40'
    )

    # %%
    # Пример запуска: главный multi-scale pre-head слой перед Detect
    result_prehead_all = run_detector_conductance(
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
        result = run_detector_conductance(
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
    plt.title("Attribution absolute error vs integration steps on log scale\nimage=person.png, layer=model.22")
    plt.grid(True, which="both", alpha=0.3)
    plt.show()

    print("\nSummary:")
    for n_steps, err in zip(step_grid, abs_errors):
        print(f"n_steps={n_steps:5d} | abs_error={err:.10f}")

    # %%
