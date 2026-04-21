# %% [markdown]
# # Реализация кода из статьи Google [How Important Is a Neuron?](https://arxiv.org/abs/1805.12233)

# %% [markdown]
# ### Импорты и базовая конфигурация
# Подключаем библиотеки, включаем градиенты и определяем устройство для вычислений.
# 

# %%
import math
import gc
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from torch.func import jvp

from modules.baseline_utils import (
    DEFAULT_BLUR_SIGMA,
    build_image_baseline,
    baseline_title_fragment,
)

torch.set_grad_enabled(True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
DTYPE = torch.float32

# %% [markdown]
# ### Загрузка модели классификации
# Инициализируем YOLO-классификатор, переводим модель на нужное устройство и проверяем классы.
# 

# %%
# Загрузка классификатора
yolo = YOLO("yolo11s-cls.pt")
model = yolo.model.to(DEVICE).eval()
class_names = yolo.names

# %% [markdown]
# ### Подготовка входа и служебные утилиты
# Определяем размер изображения, функции загрузки/препроцессинга и очистки кэша backend.
# 

# %%
IMG_SIZE = 224
DEFAULT_LAYER_NAME = "model.9"

def load_image(path, img_size=IMG_SIZE, pad_value=0.0):
    """Load an image, apply letterbox-style resizing, and return both tensor and padded RGB array."""
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
    return x, canvas


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
# Добавляем вспомогательные функции для получения активаций слоя, агрегации и построения карт conductance.
# 

# %%
# Вспомогательные утилиты для layer hook, output parsing и plotting

def split_classifier_output(out):
    """Normalize classifier output into a tuple of optional probabilities and logits tensor."""
    if isinstance(out, (tuple, list)):
        if len(out) >= 2 and torch.is_tensor(out[1]):
            return out[0], out[1]
        if len(out) >= 1 and torch.is_tensor(out[0]):
            return None, out[0]
    if torch.is_tensor(out):
        return None, out
    raise TypeError(f"Не удалось интерпретировать output классификатора типа {type(out).__name__}")

def unwrap_tensor(output):
    """Extract the first tensor object from a model output container."""
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Не удалось извлечь tensor activation из output типа {type(output).__name__}")

class LayerHook:
    def __init__(self, model, layer_name):
        """Initialize a forward hook for the selected layer and allocate activation storage."""
        self.layer_name = layer_name
        self.layer_store = {}
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise KeyError(f"Слой '{layer_name}' не найден в model.named_modules()")
        self.handle = modules[layer_name].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        """Store the latest hooked layer output during forward execution."""
        self.layer_store[self.layer_name] = out

    def clear(self):
        """Clear cached layer activations from the hook storage."""
        self.layer_store.clear()

    def get(self):
        """Return the latest cached activation for the hooked layer."""
        return self.layer_store[self.layer_name]

    def remove(self):
        """Detach the forward hook from the model layer."""
        self.handle.remove()

def reduce_filter_scores(cond_tensor):
    """Aggregate conductance over non-filter axes to produce one score per filter."""
    if cond_tensor.ndim < 2:
        raise ValueError(
            f"Ожидался tensor с batch-осью и хотя бы одной feature-осью, получено shape={tuple(cond_tensor.shape)}"
        )

    per_sample = cond_tensor[0]
    if per_sample.ndim == 1:
        return per_sample

    reduce_dims = tuple(range(1, per_sample.ndim))
    return per_sample.sum(dim=reduce_dims)

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

def build_total_conductance_overlay_figure(image_np, cond_tensor, title=None):
    """Build a figure with an overlay of total spatial conductance summed over filters."""
    cond_np = cond_tensor[0].detach().cpu().numpy()

    if cond_np.ndim < 3:
        return None

    # Общая spatial conductance:
    # сумма по фильтрам |Cond_{c,h,w}|
    total_map = cond_np.sum(axis=0)
    total_map = _normalize_map(total_map)
    total_map = _resize_map_nearest(total_map, image_np.shape[:2])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np, interpolation="nearest")
    heat = ax.imshow(total_map, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
    ax.axis("off")
    ax.set_title(title or "Total conductance overlay")

    cbar = fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized conductance", rotation=90)
    return fig


def plot_total_conductance_overlay(image_np, cond_tensor, title=None):
    """Plot an image with an overlay of total spatial conductance summed over filters."""
    fig = build_total_conductance_overlay_figure(image_np, cond_tensor, title=title)
    if fig is None:
        return None
    plt.show()
    return fig


def plot_top_filter_overlays(image_np, cond_tensor, filter_scores, top_idx, top_n=5, show=True):
    """Plot conductance overlays for the top-ranked filters on the input image."""
    cond_np = cond_tensor[0].detach().cpu().numpy()

    if cond_np.ndim < 3:
        return []

    if top_n == -1:
        selected_idx = top_idx.tolist()
    else:
        top_n = min(top_n, len(top_idx))
        selected_idx = top_idx[:top_n].tolist()
    figures = []

    for rank, idx in enumerate(selected_idx, start=1):
        fmap = cond_np[idx]
        fmap = _normalize_map(fmap)
        fmap = _resize_map_nearest(fmap, image_np.shape[:2])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image_np, interpolation="nearest")
        ax.imshow(fmap, cmap="seismic", vmin=-1.0, vmax=1.0, alpha=0.45, interpolation="nearest")
        ax.set_title(f"Filter {idx} conductance = {float(filter_scores[idx]):+.6f} (rank {rank})")
        ax.axis("off")
        figures.append(fig)

        if show:
            plt.show()

    return figures

# %% [markdown]
# Данный подход предложен в статье [How important is a neuron?](https://arxiv.org/pdf/1805.12233), идея использовать цепное правило для получение результата по промежуточному слою взята из статьи [Improving Adversarial Transferability via Neuron Attribution-Based Attacks](https://arxiv.org/pdf/2204.00008)/
# #### Вводные:
# У нас есть:
# - изображение $x \in \mathbb{R}^n$ - выпрямленная в вектор картинка. В базовом случае рассматривается grayscale, на практике просто нужно будет помнить, что $x_i \in \mathbb{R}^3 \; \forall x_i \in x$, а любые производные по $x_i$ 
# - $F(x)$ - модель-классификатор, которая изображению соотносит уверенность в некотором классе. При этом в данном случае мы считаем за $F(x)$ максимальный логит уверенности в классе (в общем случае классификатор выдаёт распределение вероятностей классов, мы лишь извлекаем наибольшее значение)
# 
# #### Постановка задачи 
# Мы хотим каждому нейрону на промежуточном слое модели $y$ сопоставить некую важность. Чем важнее нейрон, тем больший вклад в итоговое предсказание $F(x)$ он даёт, причем вклад может быть как положительным, так и отрицательным.
# 
# #### Алгоритм
# Получается, мы хотим обосновать через модель и картинку, как сложилось число $F(x)$.
# 
# Искусственно введем $x'\text{ | }F(x')\approx0$ - некую картинку, на которой предсказание модели примерно равно нулю. Достаточно хорошо этому условию удовлетворяет черная картинка, при этом точность в равенстве нулю не настолько важна, ибо с данного момента мы будем стараться обосновать разность $F(x) - F(x')$.
# 
# Сама постановка такой задачи сразу напоминает нам о правиле Ньютона-Лейбница, которое говорит:
# $$
# F(x)-F(x')
# =\int_{x'}^x F'(x)dx
# $$
# $x$ и $x'$ это векторы, по ним интегрировать будет сложно, так что перейдем к скалярной переменной интегрирования. Введем $\alpha\in[0;1]$ и скажем, что $x(\alpha)=x'+\alpha(x-x')$, при этом $\frac{dx(\alpha)}{d\alpha}=x-x'.$ Так мы через один параметр задали наши $x$ и $x'$.
# 
# Тогда 
# $$
# F(x)-F(x')
# =\int_{x'}^x F'(x)dx
# =\int_0^1 \nabla F(x(\alpha))^\top x'(\alpha) d\alpha
# =\int_0^1 \nabla F(x(\alpha))^\top (x-x') d\alpha
# $$
# 
# При этом помним, что 
# $$
# \nabla F(x(\alpha)) =
# \begin{pmatrix}
# \frac{\partial F(x(\alpha))}{\partial x_1}\\
# \frac{\partial F(x(\alpha))}{\partial x_2}\\
# \vdots\\
# \frac{\partial F(x(\alpha))}{\partial x_n}
# \end{pmatrix},
# \;
# x-x' =
# \begin{pmatrix}
# x_1-x'_1\\
# x_2-x'_2\\
# \vdots\\
# x_n-x'_n
# \end{pmatrix}.
# $$
# 
# Раскрываем скалярное произведение и получаем
# $$
# F(x)-F(x')
# =\int_0^1\sum_{i=1}^{n}\frac{\partial F(x(\alpha))}{\partial x_i}(x_i-x'_i)d\alpha.
# $$
# 
# Пользуясь линейностью интеграла запишем
# $$
# F(x)-F(x')
# =\sum_{i=1}^{n}(x_i-x'_i)\int_0^1\frac{\partial F(x(\alpha))}{\partial x_i},d\alpha.
# $$
# 
# И таким образом мы получили разложение разности $F(x)-F(x')$ по пикселям изображения $x_i$! 
# 
# Запишем важность каждого пикселя как
# $$
# \mathrm{IG}_i(x)
# :=
# (x_i-x'_i)\cdot\int_0^1 \frac{\partial F\left(x' + \alpha(x-x')\right)}{\partial x_i},d\alpha.
# $$
# Основная статья на этом заканчивается, мы же дополнительно сделаем следующий шаг через цепное правило в многомерном случае:
# $$
# A_y
# :=
# \sum_{j=1}^{m}
# \sum_{i=1}^{n} (x_i-x’_i)
# \int_0^1
# \frac{\partial F}{\partial y_j}\bigl(y(x(\alpha))\bigr)
# \frac{\partial y_j}{\partial x_i}\bigl(x(\alpha)\bigr)
# d\alpha.
# $$
# где на слое y располагается m нейронов.
# 
# Снова запишем отдельно важность каждого нейрона:
# $$
# A_{y_j}
# :=
# \sum_{i=1}^{n} (x_i-x’_i)
# \int_0^1
# \frac{\partial F}{\partial y_j}\bigl(y(x(\alpha))\bigr)
# \frac{\partial y_j}{\partial x_i}\bigl(x(\alpha)\bigr)
# ,d\alpha.
# $$
# Эта формула и используется в данном коде.
# 
# #### Особенности реализации кодом
# ##### Подсчет через якобиан
# В торче есть функция - jvp, которая позволяет посчитать произведение якобиана функции на вектор направления (Jacobain-vector product для функции $y(x)$ в точке $x_\alpha$ по направлению $\delta_x$)
# ###### Вводные
# У нас есть:
# - Слой $y(x): \mathbb{R}^d \to \mathbb{R}^m$. Оборачиваем его в функцию act_only
# - x_alpha — точка, в которой берётся производная: $x_\alpha = x_0 + \alpha(x-x_0)$
# - delta_x — направление, по которому берётся производная: $\delta_x = x-x_0$
# 
# ###### Что происходит в функции
# Для векторной функции $y:\mathbb{R}^d\to\mathbb{R}^m$ её якобиан:
# $$
# J_y(x)=
# \begin{pmatrix}
# \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_d}\\
# \vdots & \ddots & \vdots\\
# \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_d}
# \end{pmatrix}.
# $$
# Тогда произведение якобиана на вектор направления \delta_x даёт вектор:
# $$
# J_y(x_\alpha)\delta_x
# =
# \begin{pmatrix}
# \sum_i \frac{\partial y_1}{\partial x_i}(x_\alpha)\,\delta x_i\\
# \vdots\\
# \sum_i \frac{\partial y_m}{\partial x_i}(x_\alpha)\,\delta x_i
# \end{pmatrix}.
# $$
# Так как у нас
# $$
# \delta x_i = x_i-x_i',
# $$
# то для каждого нейрона $y_j$:
# $$
# \bigl(J_y(x_\alpha)\delta_x\bigr)_j
# =
# \sum_i \frac{\partial y_j}{\partial x_i}(x_\alpha)(x_i-x_i').
# $$

# %%
def compute_conductance(
    model,
    hook,
    x,
    x0,
    target_class,
    n_steps=64,
    fd_eps=1e-3,
    clear_every=8,
    warn_on_fallback=False,
):
    """Estimate layer/neuron conductance along the baseline path using midpoint integration."""
    x = x.contiguous()
    x0 = x0.contiguous()
    delta_x = (x - x0).contiguous()

    def forward_with_layer(x_in):
        """Run a forward pass and return both model output and hooked layer activation."""
        hook.clear()
        out = model(x_in)
        act = unwrap_tensor(hook.get())
        return out, act

    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device, dtype=x.dtype)
    step = 1.0 / n_steps

    cond_accum = None
    used_fallback = False

    for k in range(n_steps):
        alpha = (alphas[k] + alphas[k + 1]) / 2.0
        x_alpha = (x0 + alpha * delta_x).contiguous().detach().requires_grad_(True)

        out, act = forward_with_layer(x_alpha)
        _, logits = split_classifier_output(out)
        score = logits[0, target_class]

        # ∂F/∂y
        grad_y = torch.autograd.grad(score, act, retain_graph=False, create_graph=False)[0]

        try:
            def act_only(inp):
                """Return only layer activations for Jacobian-vector product computation."""
                inp = inp.contiguous()
                _, a = forward_with_layer(inp)
                return a

            # dy/dα
            _, act_tangent = jvp(act_only, (x_alpha,), (delta_x,))

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

            # dy/dα ≈ [y(α+ε) - y(α-ε)] / (2ε)
            act_tangent = (act_plus - act_minus) / denom
            del act_plus, act_minus

        integrand = grad_y * act_tangent
        cond_accum = integrand.detach() if cond_accum is None else (cond_accum + integrand.detach())

        del x_alpha, out, act, logits, score, grad_y, act_tangent, integrand
        hook.clear()

        if (k + 1) % clear_every == 0:
            clear_backend_cache()

    cond = cond_accum * step
    clear_backend_cache()

    if used_fallback and warn_on_fallback:
        warnings.warn(
            "jvp на текущем backend не сработал, для dy/dalpha использована конечная разность.",
            stacklevel=2,
        )

    return cond.detach()

# %% [markdown]
# ### Единый pipeline эксперимента
# Собираем полный процесс: предсказание, conductance, ранжирование фильтров и построение графиков.
# 

# %%
def run_conductance_pipeline(
    image_path,
    layer_name=DEFAULT_LAYER_NAME,
    n_steps=64,
    baseline_mode="zero",
    baseline_rgb=None,
    baseline_blur_sigma=DEFAULT_BLUR_SIGMA,
    top_n=5,
    fd_eps=1e-3,
    clear_every=8,
    verbose=False,
    show_total_plot=True,
    show_filter_plots=False,
    warn_on_fallback=False,
):
    """Execute the full conductance workflow and return metrics plus intermediate tensors."""
    x, img_np = load_image(image_path)
    x0, baseline_info = build_image_baseline(
        x,
        img_np,
        mode=baseline_mode,
        baseline_rgb=baseline_rgb,
        blur_sigma=baseline_blur_sigma,
    )

    hook = LayerHook(model, layer_name)

    try:
        def forward_with_layer(x_in):
            """Run a forward pass and return both model output and hooked layer activation."""
            hook.clear()
            out = model(x_in)
            act = unwrap_tensor(hook.get())
            return out, act

        with torch.no_grad():
            out, act = forward_with_layer(x)
            probs, logits = split_classifier_output(out)
            target_class = int(logits[0].argmax().item())
            target_name = class_names[target_class]
            target_logit = float(logits[0, target_class].item())
            target_prob = float(torch.softmax(logits[0], dim=0)[target_class].item())

        cond_tensor = compute_conductance(
            model=model,
            hook=hook,
            x=x,
            x0=x0,
            target_class=target_class,
            n_steps=n_steps,
            fd_eps=fd_eps,
            clear_every=clear_every,
            warn_on_fallback=warn_on_fallback,
        )

        filter_scores = reduce_filter_scores(cond_tensor)
        layer_score = cond_tensor.sum()

        topk = min(10, filter_scores.numel())
        top_vals, top_idx = torch.topk(filter_scores, k=topk)

        with torch.no_grad():
            out_x, _ = forward_with_layer(x)
            out_x0, _ = forward_with_layer(x0)

            _, logits_x = split_classifier_output(out_x)
            _, logits_x0 = split_classifier_output(out_x0)

            fx = float(logits_x[0, target_class].item())
            fx0 = float(logits_x0[0, target_class].item())
            abs_error = abs((fx - fx0) - float(layer_score.item()))

        total_plot_title = "\n".join(
            [
                "Total conductance",
                f"layer={layer_name}",
                f"class={target_name}",
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
            print(
                "baseline:",
                baseline_title_fragment(
                    baseline_info["baseline_mode"],
                    baseline_rgb=baseline_info["baseline_rgb"],
                    blur_sigma=baseline_info["baseline_blur_sigma"],
                ),
            )
            print("layer activation shape:", tuple(act.shape))
            print("cond tensor shape:", tuple(cond_tensor.shape))
            print("filter_scores shape:", tuple(filter_scores.shape))

            print("\ntop filters by |conductance|:")
            for rank, idx in enumerate(top_idx.tolist(), start=1):
                val = float(filter_scores[idx].item())
                print(f"{rank:2d}. filter {idx:4d}: {val:+.6f}")

            print("\nlayer conductance sum:", float(layer_score.item()))
            print("example neuron conductance:", float(cond_tensor.reshape(-1)[0].item()))
            print("example filter conductance:", float(filter_scores[0].item()))
            print("F(x)            =", fx)
            print("F(x0)           =", fx0)
            print("F(x) - F(x0)    =", fx - fx0)
            print("sum conductance =", float(layer_score.item()))
            print("abs error       =", abs_error)

        if show_total_plot:
            plot_total_conductance_overlay(
                img_np,
                cond_tensor,
                title=total_plot_title,
            )
        if show_filter_plots and top_n != 0:
            plot_top_filter_overlays(
                img_np,
                cond_tensor,
                filter_scores,
                top_idx,
                top_n=top_n,
                show=True,
            )

        result = {
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
            "baseline_mode": baseline_info["baseline_mode"],
            "baseline_rgb": baseline_info["baseline_rgb"],
            "baseline_blur_sigma": baseline_info["baseline_blur_sigma"],
            "image_np": img_np,
            "total_plot_title": total_plot_title,
        }
        return result

    finally:
        hook.remove()
        clear_backend_cache()

# %% [markdown]
# ## Эксперименты

if __name__ == "__main__":
    # %%
    # Для классификатора предпоследний перед головой слой - 9
    # Он слишком большой, 5 даёт достаточно хорошие признаки
    result = run_conductance_pipeline(
        image_path="data/jess.jpg",
        layer_name='model.5',
        n_steps=8192,
        top_n=5,
    )

    # %%
    result = run_conductance_pipeline(
        image_path="data/zebra.jpg",
        layer_name='model.5',
        n_steps=128,
        top_n=0,
    )

    # %%
    result = run_conductance_pipeline(
        image_path="data/fox.jpg",
        layer_name='model.6',
        n_steps=128,
        top_n=5,
    )

    # %%
    result = run_conductance_pipeline(
        image_path="data/fox.jpg",
        layer_name='model.6',
        n_steps=2048,
        top_n=5,
    )

    # %% [markdown]
    # ### Запуск эксперимента: `fox.jpg` (слой `model.9`)
    # Сравниваем результаты на более глубоком слое и расширенном top-N фильтров.
    # 

    # %%
    result = run_conductance_pipeline(
        image_path="data/fox.jpg",
        layer_name='model.9',
        n_steps=128,
        top_n=25,
    )

    # %% [markdown]
    # ### Запуск эксперимента: `lighthouse.jpg` (слой `model.6`)
    # Оцениваем важность нейронов для сцены с маяком на среднем уровне сети.
    # 

    # %%
    result = run_conductance_pipeline(
        image_path="data/lighthouse.jpg",
        layer_name='model.6',
        n_steps=128,
        top_n=5,
    )

    # %%
    result = run_conductance_pipeline(
        image_path="data/lighthouse.jpg",
        layer_name='model.6',
        n_steps=2048,
        top_n=5,
    )

    # %% [markdown]
    # ### Запуск эксперимента: `lighthouse.jpg` (слой `model.0`)
    # Смотрим conductance на раннем слое, чтобы сравнить с более глубокими представлениями.
    # 

    # %%
    result = run_conductance_pipeline(
        image_path="data/lighthouse.jpg",
        layer_name='model.0',
        n_steps=128,
        top_n=5,
    )

    # %% [markdown]
    # ## Проверка зависимости значения ошибки от количества шагов интегрирования
    # 
    # Цель - показать сходимость ошибки теоретического и практического conductance при увеличении числа шагов интегрирования

    # %%
    step_grid = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    abs_errors = []
    results_by_steps = {}

    for n_steps in step_grid:
        print(f"\n=== n_steps = {n_steps} ===")
        result = run_conductance_pipeline(
            image_path="data/jess.jpg",
            layer_name="model.5",
            n_steps=n_steps,
            top_n=0,
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

    # %%
    step_grid = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    abs_errors = []
    results_by_steps = {}

    for n_steps in step_grid:
        print(f"\n=== n_steps = {n_steps} ===")
        result = run_conductance_pipeline(
            image_path="data/fox.jpg",
            layer_name="model.5",
            n_steps=n_steps,
            top_n=0,
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
