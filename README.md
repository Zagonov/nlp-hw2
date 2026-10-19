# HW2: генерация кода

## Установка

```bash
cd /Users/dmitry/hse/nlp/hw-2
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Препроцессинг

Сначала рекомендуется проверить обработку на небольшом фрагменте:

```bash
python preprocess_dataset.py \
  --limit 10000 \
  --num-proc 8 \
  --output data/train_clean_test
```

Полный запуск:

```bash
python preprocess_dataset.py \
  --num-proc 8 \
  --cache-dir data/huggingface_cache \
  --output data/train_clean
```

Скрипт удаляет комментарии и docstring из поля `code`. Прогресс, скорость и ETA отображаются в терминале.

## Токенизатор

Быстрая проверка:

```bash
python tokenizer.py \
  --input data/train_clean_test \
  --limit 10000 \
  --source-vocab-size 1000 \
  --target-vocab-size 1000 \
  --min-frequency 2 \
  --source-output source_tokenizer_test.json \
  --target-output target_tokenizer_test.json
```

Обучение итогового токенизатора:

```bash
python tokenizer.py \
  --input data/train_clean \
  --source-vocab-size 4096 \
  --target-vocab-size 8192 \
  --min-frequency 10 \
  --source-output source_tokenizer.json \
  --target-output target_tokenizer.json
```

Source-токенизатор обучается только на поле `question`, target-токенизатор — только на очищенном поле `code`.

## Проверка обучения

```bash
python train.py \
  --dataset data/train_clean_test \
  --source-tokenizer source_tokenizer_test.json \
  --target-tokenizer target_tokenizer_test.json \
  --batch-size 2 \
  --gradient-accumulation 1 \
  --d-model 64 \
  --n-heads 4 \
  --d-ff 128 \
  --encoder-layers 1 \
  --decoder-layers 1 \
  --max-source-length 128 \
  --max-target-length 128 \
  --validation-size 100 \
  --validation-batches 5 \
  --num-workers 0 \
  --max-steps 10 \
  --output checkpoints/test
```

## Обучение на RTX 4090

На Vast.ai нужно перенести исходники проекта, каталог `data/train_clean`, `source_tokenizer.json` и `target_tokenizer.json`, после чего установить зависимости:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
comet login
```

Запуск обучения:

```bash
bash run_train.sh
```

Эквивалентная полная команда:

```bash
python train.py \
  --dataset data/train_clean \
  --source-tokenizer source_tokenizer.json \
  --target-tokenizer target_tokenizer.json \
  --batch-size 80 \
  --gradient-accumulation 2 \
  --d-model 640 \
  --n-heads 10 \
  --d-ff 2560 \
  --encoder-layers 8 \
  --decoder-layers 8 \
  --epochs 1 \
  --learning-rate 3e-4 \
  --max-source-length 192 \
  --max-target-length 256 \
  --validation-size 10000 \
  --validation-batches 100 \
  --num-workers 2 \
  --comet-project hw2-code-generation \
  --comet-name transformer-100m \
  --log-every 10 \
  --checkpoint-every 1000 \
  --archive-every 10000 \
  --output checkpoints/transformer-100m
```

Параметры `batch-size=80`, `gradient-accumulation=2` и `num-workers=2` подобраны
бенчмарком на RTX 4090 48 ГБ. Если возникает CUDA out of memory, нужно уменьшить
`--batch-size` и пропорционально увеличить `--gradient-accumulation`.

Модель обучается в BF16 на RTX 4090. На видеокартах без BF16 используется FP16, на CPU смешанная точность отключается.

Во время обучения `latest.pt` перезаписывается каждые `checkpoint-every` шагов.
Каждые `validate-every` шагов (по умолчанию 1000) считается validation loss.
Если он улучшился, обновляются полный `best.pt` и облегчённый
`inference-best.pt`; облегчённая лучшая версия также загружается в Comet с номером
шага в имени.
Если validation loss не улучшается в течение `early-stopping-patience` проверок,
обучение завершается досрочно, сохраняя лучшую найденную модель. При необходимости
минимально значимое улучшение можно задать через `early-stopping-min-delta`;
по умолчанию оно равно нулю. Норма градиента логируется как `train/gradient_norm`, а
нечисловая норма немедленно останавливает запуск до повреждения следующих
checkpoints.
На каждом validation-шаге для фиксированных примеров из train и validation
выполняется autoregressive-генерация. Comet получает таблицу
`generation-samples-step-XXXXXX.csv` с prompt, reference, prediction и признаком
точного токенного совпадения, а также метрики `generation/train_exact_match` и
`generation/validation_exact_match`. Число примеров каждого split задаётся через
`--logged-generation-samples` (по умолчанию 4).
Дополнительно каждые `archive-every` шагов сохраняется архивный checkpoint вида
`step-010000.pt` и облегчённый checkpoint для инференса вида
`inference-step-010000.pt`. При включённом онлайн-режиме Comet облегчённые
checkpoints автоматически загружаются в Comet. Отключить их загрузку можно
флагом `--no-upload-inference-checkpoints`. После завершения обучения также
создаётся `inference.pt`, содержащий только веса модели, конфигурацию и номер
шага; `model.pt` по-прежнему содержит полное состояние для продолжения обучения:
модель, optimizer, scheduler, GradScaler, конфигурацию, шаг, число просмотренных
примеров и ключ эксперимента Comet.
Финальные `model.pt` и `inference.pt` загружаются в Comet как `training-final` и
`inference-final` соответственно.

Рабочий запуск использует learning rate `5e-5`, warmup 5% и отдельный каталог
`checkpoints/transformer-100m-lr5e5`, чтобы не продолжать обучение из
деградировавшего checkpoint старого запуска с learning rate `3e-4`.

Продолжение прерванного запуска выполняется той же командой и с теми же параметрами модели, но с дополнительным аргументом:

```bash
bash run_train.sh --resume checkpoints/transformer-100m/latest.pt
```

В Comet записываются `train/loss`, `train/lr`, `train/optimizer_step`, `train/samples_seen`, скорость обработки примеров и `validation/loss`. При продолжении обучения логирование идёт в тот же эксперимент.

Для локального запуска без отправки данных:

```bash
python train.py \
  --dataset data/train_clean_test \
  --source-tokenizer source_tokenizer_test.json \
  --target-tokenizer target_tokenizer_test.json \
  --d-model 64 \
  --n-heads 4 \
  --d-ff 128 \
  --encoder-layers 1 \
  --decoder-layers 1 \
  --max-steps 10 \
  --num-workers 0 \
  --comet-mode offline \
  --output checkpoints/test
```

Offline-эксперимент сохраняется в подкаталоге `comet` внутри каталога эксперимента. Его можно отправить позднее:

```bash
comet upload checkpoints/test/comet/*.zip
```

Для полного отключения логирования используется `--comet-mode disabled`. Если проект находится не в workspace по умолчанию, нужно добавить `--comet-workspace имя`.

### Проверка переобучения на одном батче

Перед полным запуском можно проверить весь путь обучения и autoregressive
генерации на четырёх фиксированных примерах:

```bash
source /venv/main/bin/activate
python one_batch_test.py
```

Тест использует ту же архитектуру размером 100M, обучает её только на одном батче
и завершается успешно, если loss опустился до 0.05 или ниже и все четыре программы
совпали с эталонными на уровне токенов. Подробный отчёт и сгенерированный код
сохраняются в `one_batch_test/report.json`.

## Проверка скорости инференса

Локальный запуск на необученной модели:

```bash
python inference.py \
  --num-examples 512 \
  --batch-size 64 \
  --source-length 128 \
  --max-new-tokens 64
```

Для проверки на Kaggle используется `kaggle_inference_benchmark.ipynb`. Нужно создать Kaggle Notebook с T4 GPU и добавить `model.py`, `inference.py`, `source_tokenizer.json` и `target_tokenizer.json` как Input Dataset либо загрузить их в `/kaggle/working`.

Ноутбук скачивает настоящий split `test` датасета `SlavaYus/tinysms_qa_12m`, прогоняет все 1919 промптов с лимитом 64 токена и сохраняет ответы в `/kaggle/working/predictions.jsonl`. В файле находится ровно по одному объекту `{"code": "<вывод модели>"}` на задачу, без дополнительных полей. Замер включает токенизацию, генерацию и декодирование, показывает скорость, peak GPU memory и прохождение лимита 180 секунд.

Инференс обученного checkpoint:

```python
from inference import generate_texts, load_model
from tokenizers import Tokenizer

model = load_model("checkpoints/transformer-100m/model.pt")
source_tokenizer = Tokenizer.from_file("source_tokenizer.json")
target_tokenizer = Tokenizer.from_file("target_tokenizer.json")

predictions = generate_texts(
    model,
    source_tokenizer,
    target_tokenizer,
    prompts,
    batch_size=64,
    max_new_tokens=64,
)
```
