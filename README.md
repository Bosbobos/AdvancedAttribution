# Ноутбуки проекта

Ниже перечислены все основные ноутбуки репозитория. Они разделены на две группы:

- вспомогательные ноутбуки: реализации методов, переносы на detector и служебные ноутбуки сравнения;
- основные экспериментальные ноутбуки: отдельные серии экспериментов, перечисленные в порядке создания файлов.

## Вспомогательные ноутбуки

### `HowImportantIsANeuron.ipynb`
Реализация метода conductance из статьи *How Important Is a Neuron?* для классификатора YOLO. В ноутбуке есть:

- базовая теория и вывод формулы conductance;
- реализация через хуки и якобиан по выбранному слою;
- единый pipeline для запуска эксперимента на одном изображении;
- запуски на примерах;
- сравнение нескольких слоёв;
- отдельная проверка зависимости ошибки от числа шагов интегрирования.

### `HowImportantIsANeuron_detector.ipynb`
Перенос conductance-подхода на детектор YOLO. В ноутбуке есть:

- выбор целевого предсказания детектора через query/class/bbox-логику;
- сравнение схем выбора бокса, включая raw top-1 и class-aware NMS;;
- эксперименты на конкретном слое перед головой и всех предголовных слоях;
- проверка сходимости ошибки по числу шагов интегрирования.

### `ImprovingAdvTransViaAttrib.ipynb`
Базовая реализация метода NAA из статьи *Improving Adversarial Transferability via Neuron Attribution-Based Attacks* для классификатора. В ноутбуке есть:

- загрузка и подготовка классификатора;
- реализация вычисления neuron attribution через выбранный слой;
- единый pipeline для получения итоговой карты атрибуции;
- эксперименты и сравнения слоёв;
- проверка сходимости ошибки;

### `ImprovingAdvTransViaAttrib_detector.ipynb`
Перенос NAA на detector. В ноутбуке есть:

- подготовка detector-таргета и ROI-агрегации;
- реализация attribution pipeline для detector-выходов;
- запуски на нескольких изображениях;
- проверка сходимости ошибки.

### `MethodComparison.ipynb`
Служебный ноутбук для side-by-side сравнения IG и NAA на одинаковых параметрах. Он использует код из `modules/MethodComparison.py` и делает:

- сравнение IG и NAA для классификаторов;
- сравнение IG и NAA для детекторов;
- единый экспорт итоговых картинок в Obsidian-совместимом markdown-формате;
- общий экспорт четырёх готовых таблиц: для двух classifier-конфигураций и двух detector-конфигураций.

### `Benchmarks/AOPCBenchmark.ipynb`
Служебный ноутбук для простого quality-benchmark по `AOPC` на `Oxford Pets`. Он использует код из `modules/aopc_benchmark.py` и делает:

- сравнение `IG`, `NAA` и нескольких конфигураций `Cheap-IG` на `100` изображениях из `oxford_pets`;
- текущую grid-конфигурацию `Cheap-IG` на сегменте `[0, 0.2]` с positive-only отбором `top_k = 8000 / 16000 / 32000`;
- переключаемый hybrid-fill режим `fill_mode="naa_scaled"` с отдельными прогонами для `rho = 0.8` и `rho = 1.0`;
- classifier-benchmark на фиксированном слое без `NAOPC`-нормализации;
- режим `unit_mode="neuron"` по умолчанию и возможность переключиться на `unit_mode="spatial_cell"` или `unit_mode="filter"`;
- режим `perturbation_mode="both"` по умолчанию и возможность переключиться на `deletion` или `insertion`;
- режим `budget_mode="percent_steps"` по умолчанию: `1%` юнитов за шаг, `100` cumulative steps;
- ручной override budget-сетки через `perturbation_counts` или `perturbation_fractions`;
- сохранение кэшей, summary-json, markdown-отчёта и графиков в `output/`;
- отображение агрегированного markdown-отчёта прямо в ноутбуке.

### `Benchmarks/ROADBenchmark.ipynb`
Служебный ноутбук для classifier-only benchmark по `ROAD` в режиме `MoRF` на `Oxford Pets`. Он использует код из `modules/road_benchmark.py` и делает:

- сравнение `IG`, `NAA` и полной grid-конфигурации `Cheap-IG` на `100` изображениях из `oxford_pets`;
- текущую grid-конфигурацию `Cheap-IG` на сегменте `[0, 0.2]` с positive-only отбором `top_k = 8000 / 16000 / 32000`;
- отдельные варианты `Cheap-IG` без хвоста и с hybrid-fill хвостом `NAA` для `rho = 0.8` и `rho = 1.0`;
- classifier-benchmark на фиксированном слое через классический `ROAD MoRF` с `Noisy Linear Imputation`;
- использование clean top-1 предсказания `yolo11s-cls` как target-класса;
- primary score через `target logit drop AOC`, где больше значит лучше;
- secondary диагностику через `top-1 consistency`;
- фиксированные ROAD-процентили `10, 20, ..., 90`;
- сохранение кэшей, summary-json, markdown-отчёта, grouped preview tables и графиков в `output/`;
- отображение агрегированного markdown-отчёта прямо в ноутбуке.

### `Benchmarks/ROADHyperparamSearch.ipynb`
Служебный ноутбук для staged-search гиперпараметров `Cheap-IG` по `ROAD MoRF` на `Oxford Pets`. Он использует код из `modules/road_hparam_search.py` и делает:

- поиск гиперпараметров только для семейства `Cheap-IG`, а `IG` и `NAA` использует только на финальной holdout-оценке;
- детерминированный split первых `100` изображений из `oxford_pets` на `70` search и `30` holdout;
- текущий search space по `top_k = 4000 / 6000 / 8000 / 10000 / 16000 / 32000`, `segment_end = 0.1 / 0.12 / 0.15 / 0.2`, `fill_mode = zero / naa_scaled`, `rho = 0.6 / 0.8 / 1.0 / 1.2` и `n_steps = 24 / 48 / 96 / 192`;
- staged-search вместо полного grid-search: topology screen, `rho` sweep и refinement по `n_steps`;
- подобранную сетку `n_steps`, где число реальных alpha-точек на сегменте действительно меняется между конфигурациями, а не остаётся почти одинаковым;
- Pareto-отбор по `target_logit_drop_aoc` и `runtime_s`;
- финальный выбор трёх конфигураций `best_quality`, `best_balanced` и `fastest_pareto`;
- отдельные markdown/json-отчёты, csv-таблицу кандидатов и набор графиков для search и holdout;
- отображение search- и holdout-отчётов прямо в ноутбуке.

### `Benchmarks/AlphaSegmentLatentAOPC.ipynb`
Служебный ноутбук для alpha-segment sweep по latent-space `AOPC` на `Oxford Pets`. Он использует код из `modules/alpha_segment_benchmark.py` и делает:

- сравнение `IG`, `NAA` и одного фиксированного tuned `Cheap-IG` на первых `100` изображениях из `oxford_pets`;
- sweep правой границы alpha-отрезка `segment_end = 0.1, 0.2, ..., 1.0` при фиксированном `n_steps = 192`;
- raw-neuron benchmark в latent space с `deletion` only и score `target logit drop AOC@20%`;
- сравнение пяти latent donors / imputers: `zero_baseline`, `black_act`, `blur_act`, `layer_mean_exclusive`, `spatial_nli_same_channel`;
- агрегированные кривые `mean ± std` по датасету, single-image diagnostics и donor-vs-segment heatmaps;
- peak-summary таблицу с `score@0.1`, `best_end`, `peak_contrast` и средними по всем segment-end;
- donor-independent visual tables по `10` изображениям и `10` значениям `segment_end` для `IG`, `NAA` и `Cheap-IG`;
- сохранение кэшей, markdown/json-отчёта и набора графиков в `output/`;
- отображение агрегированного markdown-отчёта прямо в ноутбуке.

### `Benchmarks/LatentBaselineBenchmark.ipynb`
Служебный ноутбук для classifier-only latent benchmark без `ROAD`-импутации. Он использует код из `modules/latent_baseline_benchmark.py` и делает:

- сравнение `IG`, `NAA` и полной grid-конфигурации `Cheap-IG` на `100` изображениях из `oxford_pets`;
- использует те же `Cheap-IG` варианты, что и `Benchmarks/ROADBenchmark.ipynb`: `top_k = 8000 / 16000 / 32000`, `segment=[0,0.2]`, positive-only, без хвоста и с `naa_scaled` хвостом для `rho = 0.8 / 1.0`;
- заменяет `ROAD`-импутацию на latent donors / baselines, где порядок по умолчанию начинается с `black_act`;
- считает raw-neuron `deletion`-benchmark со score `target logit drop AOC@20%` по budget-сетке `1..20%`;
- строит road-like summary, distributions, deletion curves, pairwise win-rate heatmaps и visual preview tables;
- сохраняет кэши, markdown/json-отчёт и графики в `output/`;
- отображает агрегированный markdown-отчёт прямо в ноутбуке.

### `Benchmarks/NeuronShapleyBenchmark.ipynb`
Служебный ноутбук для audit-benchmark на реальной модели и реальном датасете через `Monte-Carlo Shapley` oracle. Он использует код из `modules/neuron_shapley_benchmark.py` и делает:

- сравнение `IG`, `NAA` и одного фиксированного варианта `Cheap-IG+[0,0.1]/k8000/zero`;
- classifier-only benchmark на слое `model.6` с target = clean top-1 logit;
- default `unit_mode=spatial_cell` и полный `14x14` spatial-cell pool размера `196` для `model.6`;
- `128` случайных перестановок для Monte-Carlo Shapley с возможностью уменьшить budget для smoke-run;
- default pool selection `active_random` с возможностью переключиться на `stratified_activation_change`;
- default oracle imputer `black_act`;
- primary метрику `Spearman` с oracle и secondary метрики `NDCG@10`, `Recall@10`, `Sign agreement`;
- markdown/json-отчёт, pairwise win-rate heatmap и visual preview в `output/`;
- отображение агрегированного markdown-отчёта, leaderboard-таблицы, pairwise matrix, core/oracle diagnostics и figure gallery прямо в ноутбуке.

### `Benchmarks/FeatureSelectionBenchmark.ipynb`
Служебный ноутбук для classifier-only benchmark по мотивам секции `5.2 Feature Selection Study` из *How Important Is a Neuron?*. Он использует код из `modules/feature_selection_benchmark.py` и делает:

- сравнение `IG`, `NAA` и одного фиксированного варианта `Cheap-IG+[0,0.1]/k8000/zero`;
- работает на фильтрах/каналах слоя `model.6`, а не на raw neurons;
- строит несколько детерминированных `5-way` задач на `Oxford Pets`, по умолчанию `cats_a`, `cats_b`, `dogs_a`, `dogs_b`;
- использует `10 train / 10 eval` изображений на класс;
- агрегирует importance по классу как `positive_mean` и выбирает global top-k filters по правилу `max_over_classes`;
- обучает `StandardScaler + LogisticRegression` probe только на `GAP`-активациях выбранных фильтров;
- репортит `accuracy` и `macro-F1` по `k`, task-level heatmaps и overlap выбранных фильтров между методами;
- сохраняет markdown/json-отчёт и figure-набор в `output/` и показывает их прямо в ноутбуке.

### `Benchmarks/ImageNetFeatureSelectionBenchmark.ipynb`
Служебный ноутбук для ImageNet-версии benchmark’а из секции `5.2 Feature Selection Study`, максимально близкой к условиям статьи. Он использует код из `modules/imagenet_feature_selection_benchmark.py` и делает:

- работает на `imagenet_val`, разложенном по `1000` synset-папкам;
- использует `4` задачи по `5` классов: две related (`dogs_related`, `cats_related`) и две random;
- использует `30 train / 20 eval` изображений на класс, как в статье;
- считает атрибуцию по `GT class logit`, а не по `argmax`;
- сравнивает `IG`, `NAA` и `Cheap-IG+[0,0.1]/k8000/zero`;
- использует `filter/channel` как unit, `positive_mean` как class aggregation и `max_over_classes` как selection rule;
- обучает `StandardScaler + LogisticRegression` probe на `GAP`-активациях выбранных фильтров;
- сохраняет markdown/json-отчёт и figure-набор в `output/` и показывает их прямо в ноутбуке.

### `Benchmarks/ImageNetFeatureSelectionHyperparamSearch.ipynb`
Служебный ноутбук для staged-search гиперпараметров `Cheap-IG` поверх ImageNet-версии benchmark’а из секции `5.2`. Он использует код из `modules/imagenet_feature_selection_hparam_search.py` и делает:

- ищет гиперпараметры только для семейства `Cheap-IG`, а `IG` и `NAA` добавляет только на финальной holdout-оценке;
- использует те же `4` задачи по `5` ImageNet-классов и детерминированно делит их на search/holdout по task-level split;
- использует тот же расширенный search space, что и `Benchmarks/ROADHyperparamSearch.ipynb`: `top_k = 4000 / 6000 / 8000 / 10000 / 16000 / 32000`, `segment_end = 0.1 / 0.12 / 0.15 / 0.2`, `fill_mode = zero / naa_scaled`, `rho = 0.6 / 0.8 / 1.0 / 1.2` и `n_steps = 24 / 48 / 96 / 192`;
- делает staged-search вместо полного grid-search: topology screen, `rho` sweep и refinement по `n_steps`;
- использует primary quality score как `mean accuracy over k`, то есть усреднённую accuracy linear probe по всем `k`, а `macro-F1` держит как secondary diagnostic;
- делает Pareto-отбор по `mean accuracy over k` и `runtime_s`;
- выбирает трёх finalists `best_quality`, `best_balanced` и `fastest_pareto`, затем сравнивает их с `IG` и `NAA` на holdout;
- сохраняет markdown/json-отчёты, csv-таблицу кандидатов и figure-набор в `output/` и показывает их прямо в ноутбуке.

### `Benchmarks/NAOPCBenchmark.ipynb`
Служебный ноутбук для quality-benchmark по `NAOPC` на `Oxford Pets`. Он использует код из `modules/naopc_benchmark.py` и делает:

- сравнение `IG`, `Cheap-IG` и `NAA` на `100` изображениях из `oxford_pets`;
- запуск classifier-benchmark на фиксированном слое и фиксированной cheap-IG конфигурации;
- режим `unit_mode="spatial_cell"` по умолчанию и возможность переключиться на `unit_mode="filter"`;
- режим `limit_mode="beam"` по умолчанию и возможность переключиться на `limit_mode="exact"` для маленьких candidate-наборов;
- сохранение кэшей, summary-json, markdown-отчёта и графиков в `output/`;
- отображение агрегированного markdown-отчёта прямо в ноутбуке.

### `MethodTimingComparison.ipynb`
Служебный ноутбук для замеров времени работы методов. В текущем виде он включает:

- поизображенческие замеры IG и NAA сначала для classifier, затем для detector;
- агрегированный summary по набору `Oxford Pets`;
- таблицу со средними и стандартными отклонениями для classifier и detector;
- grouped bar plot с error bars для сравнения IG и NAA;
- замеры `cheap-ig` на классификаторе и детекторе, где берётся ранний alpha-сегмент, а нейроны выбираются по top-k маске;
- кэширование timing-результатов и диагностику используемого устройства (`cpu/cuda/mps`).

## Дополнительные модули

### `modules/aopc_benchmark.py`
Упрощённый модуль для quality-benchmark методов атрибуции через `AOPC` на внутренних юнитах слоя. Он делает:

- единый classifier-benchmark для `IG`, `NAA` и нескольких конфигураций `cheap-IG`;
- поддержку hybrid residual fill для `cheap-IG` через `fill_mode="zero" | "naa_scaled"` и параметр `fill_rho`;
- режимы `unit_mode="neuron"`, `unit_mode="spatial_cell"` и `unit_mode="filter"`, где по умолчанию используется `neuron`;
- режимы `perturbation_mode="deletion"`, `perturbation_mode="insertion"` и `perturbation_mode="both"`, где по умолчанию используется `both`;
- режимы budget-сетки, где по умолчанию используется `budget_mode="percent_steps"` с шагом `1%` и `100` cumulative steps;
- ранжирование всех юнитов выбранного слоя без `candidate_top_k` и без `lower/upper` limits;
- perturbation на фиксированной budget-сетке через deletion/insertion на уровне юнитов слоя;
- трёхуровневое кэширование: отдельно core-метаданные, method outputs и method-evaluation curves;
- генерацию markdown-отчёта, json-summary и набора графиков в output-директории.

### `modules/road_benchmark.py`
Classifier-only модуль для quality-benchmark методов атрибуции через `ROAD MoRF`. Он делает:

- единый benchmark для `IG`, `NAA` и нескольких конфигураций `cheap-IG`;
- классический `ROAD` в image-space через `Noisy Linear Imputation` на базе `scipy`;
- проекцию internal attribution в input-space через сумму spatial attribution по каналам выбранного слоя;
- `MoRF`-ранжирование по positive-only карте с fallback на `abs`, если карта вырождается;
- использование clean top-1 предсказания как target-класса;
- primary score `target_logit_drop_aoc` и secondary `top-1 consistency`;
- фиксированные ROAD-кривые по percentiles `10..90`, scalar score `target_logit_drop_aoc` и pairwise win-rate;
- трёхуровневое кэширование: отдельно core-метаданные по clean image, отдельно method outputs с sidecar `npz` для spatial/ranking maps и отдельно method-evaluation curves;
- генерацию markdown-отчёта, json-summary, grouped visual preview tables и набора графиков в output-директории.

### `modules/road_hparam_search.py`
Модуль для staged-search гиперпараметров `Cheap-IG` поверх `ROAD MoRF`. Он делает:

- фиксированный search space по `top_k`, alpha-сегменту, `fill_mode`, `fill_rho` и `n_steps`, с явным смещением в сторону меньших `top_k`;
- детерминированный split `search/holdout` на одном и том же наборе изображений;
- staged pruning: topology screen, `rho` sweep и refinement по `n_steps`;
- хранение для каждого кандидата не только `n_steps`, но и фактического `segment_step_count` на выбранном alpha-отрезке;
- Pareto-отбор по `target_logit_drop_aoc` и `runtime_s` без дублирования логики самого ROAD benchmark;
- финальный holdout-run только для `IG`, `NAA` и трёх selected finalists;
- генерацию `search_report.md`, `search_summary.json`, `search_candidates.csv`, `holdout_report.md`, `holdout_summary.json` и поисковых графиков в output-директории.

### `modules/alpha_segment_benchmark.py`
Classifier-only модуль для alpha-segment sweep по latent-space `AOPC`. Он делает:

- сравнение segment-версий `IG`, `NAA` и одного фиксированного tuned `Cheap-IG` на raw neurons `(c,h,w)`;
- локальные wrappers для `IG-segment[0,e]` и `NAA-segment[0,e]` без изменения старых notebook entrypoints;
- classifier benchmark с `deletion` only и score `target logit drop AOC@20%`, плюс нормализованный secondary score;
- fixed grid `segment_end = 0.1, 0.2, ..., 1.0` и budget grid `1..20%` top-neurons;
- пять latent donors / imputers: `zero_baseline`, `black_act`, `blur_act`, `layer_mean_exclusive`, `spatial_nli_same_channel`;
- трёхуровневое кэширование: отдельно core-метаданные и donor-активации, отдельно method outputs с sidecar `npz` для raw neuron scores и overlay maps и отдельно method-evaluation curves на fixed budget grid;
- вычисление derived diagnostics `best_end` и `peak_contrast` для каждой пары method-family × donor;
- генерацию markdown-отчёта, json-summary и набора aggregate / single-image / visual figures в output-директории.

### `modules/latent_baseline_benchmark.py`
Classifier-only модуль для road-like latent benchmark без `ROAD`-импутации. Он делает:

- переиспользует raw-neuron latent deletion evaluator из `modules/alpha_segment_benchmark.py`, но без alpha-segment sweep;
- поддерживает `IG`, `NAA` и arbitrary grid-конфигурации `cheap-IG` через `method_spec` API;
- сравнивает несколько latent donors / baselines, при этом default donor — `black_act`;
- строит сводные `AOC20`/`AOC20_norm` heatmaps по `method × donor`;
- строит per-donor summary, per-image distributions, deletion curves и pairwise win-rate heatmaps;
- генерирует grouped visual preview tables по исходным attribution overlays без зависимости от donor;
- сохраняет markdown-отчёт, json-summary и figure-набор в output-директории.

### `modules/neuron_shapley_benchmark.py`
Classifier-only модуль для audit-benchmark нейронной атрибуции через `Monte-Carlo Shapley` oracle на реальной модели и реальном датасете. Он делает:

- переиспользует core/method cache из `modules/alpha_segment_benchmark.py`, а новым кодом добавляет независимый candidate-pool, oracle и comparison layer;
- поддерживает `IG`, `NAA` и фиксированные/произвольные конфигурации `cheap-IG` через `method_spec` API;
- поддерживает `unit_mode=neuron` и `unit_mode=spatial_cell`, при этом default-постановка сейчас `spatial_cell` с `pool_size=196`, `num_permutations=128`, `oracle_imputer_kind=black_act`;
- два режима выбора пула: `active_random` и `stratified_activation_change`;
- считает per-image `Spearman`, `NDCG@k`, `Recall@k` и `Sign agreement` между методом и Shapley oracle;
- строит summary, metric distributions, pairwise win-rate heatmap и visual preview по первым изображениям;
- сохраняет markdown-отчёт, json-summary и figure-набор в output-директории.

### `modules/feature_selection_benchmark.py`
Classifier-only модуль для feature-selection benchmark по мотивам секции `5.2` из *How Important Is a Neuron?*. Он делает:

- переиспользует core/method cache из `modules/alpha_segment_benchmark.py` и не требует нового attribution pipeline;
- поддерживает `IG`, `NAA` и arbitrary fixed-конфигурации `cheap-IG` через `method_spec` API;
- строит детерминированные `5-way` задачи по классам `Oxford Pets`;
- агрегирует channel importance per class и выбирает global top-k filters по правилу `max_over_classes`;
- извлекает `GAP`-признаки выбранных фильтров из clean activations и обучает линейный probe;
- считает `accuracy` и `macro-F1` по каждому `task × method × k`;
- строит mean curves, task heatmaps и Jaccard overlap выбранных фильтров;
- сохраняет markdown-отчёт, json-summary и figure-набор в output-директории.

### `modules/imagenet_feature_selection_benchmark.py`
Classifier-only модуль для ImageNet-версии feature-selection benchmark, максимально близкой к секции `5.2` из *How Important Is a Neuron?*. Он делает:

- использует `ImageNet val` с `30/20` split на класс;
- строит default-набор из `4` задач по `5` классов: `dogs_related`, `cats_related`, `random_a`, `random_b`;
- использует `GT class logit` через `target_class_override` в classifier attribution pipeline;
- поддерживает `IG`, `NAA` и fixed-конфигурации `cheap-IG` через `method_spec` API;
- выбирает top-k filters по aggregated class importance и обучает linear probe на `GAP`-активациях;
- считает `accuracy` и `macro-F1`, строит task heatmaps и overlap выбранных фильтров;
- предполагает стандартное соответствие `sorted wnids in imagenet_val -> classifier class index` для `ImageNet-1k`.

### `modules/naopc_benchmark.py`
Единый модуль для quality-benchmark методов атрибуции через `NAOPC` на внутренних юнитах слоя. Он делает:

- единый benchmark для classifier и detector;
- поддержку `IG`, `NAA` и нескольких конфигураций `cheap-IG` через единый `method_spec` API;
- режимы `unit_mode="spatial_cell"` и `unit_mode="filter"`, где по умолчанию используется `spatial_cell`;
- фиксированный candidate-набор юнитов на картинку, независимый от метода;
- perturbation через подмену выбранных юнитов на baseline-активации;
- вычисление `AOPC`, `NAOPC`, lower/upper limits и perturbation-кривых;
- два режима нормализации: `limit_mode="exact"` для маленьких candidate-наборов и `limit_mode="beam"` для `NAOPC_beam` из оригинальной статьи;
- трёхуровневое кэширование: отдельно method outputs, отдельно core NAOPC-ядро на картинку и отдельно method-evaluation curves на фиксированном candidate-наборе;
- генерацию markdown-отчёта, json-summary и набора графиков в output-директории.

## Основные экспериментальные ноутбуки

Ниже ноутбуки перечислены в порядке создания по файловому `birthtime`.

### `Experiments/ImprovingAdvTransViaAttrib_covariance.ipynb` — 2026-03-25
Первая отдельная серия экспериментов по ковариационной структуре NAA. В ноутбуке есть:

- проверка предположения о нулевой ковариации;
- сравнение exact и approximate neuron statistics;
- анализ связи между ковариацией, важностью нейрона и итоговым attribution;
- диагностика структуры ковариации по слоям и по пути интегрирования;
- сегментная статистика ковариации на датасете и сводные графики с error bars;
- микровыводы по тому, когда ковариацией можно пренебречь, а когда нет.

### `Experiments/ImprovingAdvTransViaAttrib_covariance_approximation.ipynb` — 2026-03-25
Основной ноутбук по приближению conductance/attribution на раннем alpha-сегменте. В нём есть несколько блоков экспериментов:

- `Exact segment vs Approx + Cov`: сравнение точного сегментного вклада с приближением через mean и covariance на раннем участке траектории;
- `Hybrid top-n`: точная коррекция только для top-n нейронов поверх дешёвой аппроксимации;
- `Нормировка exact segment`: обучение global scale на первых `train_n=50` изображениях и тестирование на следующих `test_n=50`;
- `Cheap scan -> top-k -> exact segment variants`: дешёвый предварительный скоринг, выбор top-k нейронов и точный пересчёт только на них;
- `Универсальный раннер`: единая инфраструктура для пакетного запуска и сохранения результатов;
- `Как собирается значение логита`: визуализация накопления оценки по alpha и по числу нейронов;
- `Карты важности по alpha`: сравнение full exact-карты и sparse signed top-k карты.

### `Experiments/ImprovingAdvTransViaAttrib_covariance_delta_y_alpha_0_1.ipynb` — 2026-04-07
Поздний focused-analysis ноутбук для короткого alpha-отрезка `[0, 0.1]`. В нём есть:

- сравнение `covariance[0, 0.1]` с `exact_attr[0, 0.1]`;
- сравнение `covariance[0, 0.1]` с точным `Δy(0.1)`;
- отдельные режимы анализа `abs`, `positive` и `negative`;
- Pearson и Spearman корреляции на raw-score и после `log1p`;
- overlap top-k множеств;
- capture of reference mass и отношение `capture / oracle-capture`;
- shape similarity ранговых кривых;
- набор диагностических графиков: scatter, top-k curves, binned mean curves, smoothed curves и cumulative mass curves.
