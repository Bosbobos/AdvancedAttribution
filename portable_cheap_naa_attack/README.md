# Portable Cheap-IG NAA Attack

Самодостаточная `notebook-first` директория для classifier-only атаки на `yolo11s-cls` по мотивам NAA с текущим `cheap-IG` sparse selection.

## Что внутри

- `attack_core.py` — основной API `run_attack(...)` и outer MIM-loop.
- `cheap_ig_core.py` — sparse `cheap-IG` selection и surrogate loss.
- `transforms.py` — `DIM` и `PIM`-style преобразования для режима `variant="pd"`.
- `visualization.py` — live plots, карты важности target-класса, PNG и JSON export.
- `comparison.py` — overlay-графики и side-by-side карты важности для сравнения `Cheap-IG` и `NAA`.
- `attribution_methods.py` — self-contained benchmark пяти attribution methods: `Full IG`, `Full IG` на сегменте, `NAA`, legacy `cheap-IG`, корректный `cheap-IG`, где exact IG считается только для выбранных нейронов.
- `config.py` — типизированный `AttackConfig`.
- `example_config.py` — готовые base / pd пресеты.
- `run_attack.ipynb` — основной notebook сценарий.
- `compare_cheap_ig_vs_naa.ipynb` — notebook для side-by-side сравнения нашего метода и reference `NAA`.
- `compare_attribution_methods.ipynb` — notebook для visual/time comparison пяти attribution methods на классификаторе.
- `naa_reference/` — отдельная адаптированная реализация baseline `NAA` и заметки по адаптации.
- `weights/yolo11s-cls.pt` — локальная копия весов.

Директория не импортирует код из `modules/*` и может быть вынесена отдельно.

## Быстрый старт

1. Откройте notebook:

```bash
cd portable_cheap_naa_attack
python3 -m notebook run_attack.ipynb
```

2. Или вызовите API напрямую:

```python
from portable_cheap_naa_attack import AttackConfig, run_attack

config = AttackConfig(variant="base")
result = run_attack(
    image_paths=["../data/jess.jpg"],
    config=config,
)
```

## Дефолты

- layer: `model.6`
- alpha-segment: `[0.0, 0.1]`
- selection: `signed`, `top_k=8000` на положительные и `top_k=8000` на отрицательные нейроны
- `fill_mode="zero"`
- `ia_steps=30`
- `epsilon=16/255`
- `attack_steps=10`
- `step_size=1.6/255`
- `momentum=1.0`
- `gamma=1.0`
- `variant="base"` или `variant="pd"`

## Выходы

- `outputs/latest/latest_adv.png` или `outputs/latest/<sample>.png`
- `outputs/best/best_adv.png` или `outputs/best/<sample>.png`
- `outputs/attack_history.json`
- live notebook plots с картой важности target-класса до атаки и на текущей итерации после обновления `x_adv`

## Attribution Comparison

Для отдельного сравнения attribution methods откройте:

```bash
cd portable_cheap_naa_attack
python3 -m notebook compare_attribution_methods.ipynb
```

Notebook сравнивает:

- `Full IG [0,1]`
- `Full IG [segment_start, segment_end]`
- `NAA`
- `Old Cheap-IG`
- `New Cheap-IG`

`New Cheap-IG` реализован так:

1. dense `NAA` ranking на всём пути `[0,1]`
2. выбор top-`k` нейронов
3. exact segment IG только для этих выбранных нейронов
4. optional legacy tail fill через cheap approximation

Для attribution benchmark `n_steps` и `ranking_steps` разведены отдельно:

- `n_steps` — число шагов для exact `Full IG` и segment exact `New Cheap-IG`
- `ranking_steps` — число шагов для `NAA` ranking и legacy `Old Cheap-IG`

## Замечание по cheap-IG loss

На каждой итерации sparse neuron selection пересчитывается по текущему `x_adv`. Для самого update используется устойчивый first-order surrogate на текущих detached `cheap-IG` весах выбранных нейронов. Это сохраняет sparse `cheap-IG` selection logic и делает атаку переносимой между `cuda`, `mps` и `cpu` без обязательного second-order differentiation.
