# eda-cli – CLI + HTTP API для EDA (HW04)

Проект для домашнего задания HW04 по семинару S04.
Основан на коде HW03: CLI, эвристики качества и тесты сохранены,
добавлен HTTP API на FastAPI.

---

## Структура проекта

```text
homeworks/
  HW04/
    eda-cli/
      pyproject.toml
      uv.lock
      README.md
      .gitignore
      .python-version
      data/
        example.csv
      src/
        eda_cli/
          __init__.py
          core.py
          viz.py
          cli.py
          api.py
      tests/
        test_core.py
      reports_example/   # примеры сгенерированных отчётов (не коммитятся)
      reports_after_core/
      reports_check/
```

Основной код пакета лежит в `src/eda_cli/`.

---

## Установка зависимостей

Находясь в корне проекта `eda-cli`:

```bash
cd homeworks/HW04/eda-cli
uv sync
```

`uv` создаст локальное окружение `.venv` и установит все зависимости из `pyproject.toml`.

---

## Доступные команды CLI (наследие HW03)

После установки зависимостей запуск выполняется через `uv run`:

### 1. Обзор датасета: `overview`

Краткий вывод размеров и сводной таблицы по колонкам.

```bash
uv run eda-cli overview data/example.csv
```

Что делает:

* печатает число строк и столбцов;
* показывает по каждой колонке:

  * тип данных;
  * количество и долю пропусков;
  * количество уникальных значений;
  * базовые числовые статистики (min/max/mean/std для числовых столбцов).

---

### 2. Полный отчёт: `report`

Генерация полного EDA-отчёта и артефактов в указанную папку.

Базовый пример:

```bash
uv run eda-cli report data/example.csv --out-dir reports_example
```

В результате в каталоге `reports_example` будут созданы:

* `summary.csv` – сводка по колонкам;
* `missing.csv` – таблица пропусков по колонкам (если пропуски есть);
* `correlation.csv` – корреляционная матрица для числовых признаков (если их ≥ 2);
* папка `top_categories/` – CSV с top-значениями для категориальных колонок;
* `hist_*.png` – гистограммы числовых признаков;
* `missing_matrix.png` – визуализация пропусков;
* `correlation_heatmap.png` – тепловая карта корреляций;
* `report.md` – основной markdown-отчёт.

#### Дополнительные опции `report`

Команда `report` параметризована:

```bash
uv run eda-cli report data/example.csv \
  --out-dir reports_check \
  --top-k-categories 3 \
  --max-hist-columns 4 \
  --title "Отчёт по example.csv"
```

Опции:

* `--out-dir TEXT` – каталог, куда сохраняется отчёт (по умолчанию `reports`);
* `--max-hist-columns INTEGER` – сколько числовых колонок включать в набор гистограмм
  (по умолчанию 6);
* `--top-k-categories INTEGER` – сколько top-значений сохранять для каждой категориальной колонки
  (по умолчанию 5, минимум 1);
* `--title TEXT` – заголовок markdown-отчёта (по умолчанию `"EDA-отчёт"`).

Выбранные значения `top_k_categories` и `max_hist_columns` выводятся в разделе
«Параметры отчёта» внутри `report.md`.

---

## Эвристики качества данных

Логика находится в `src/eda_cli/core.py`, функция `compute_quality_flags`.

Возвращаемый словарь `flags` включает:

Базовые метрики:

* `too_few_rows` – слишком мало строк (меньше 100);
* `too_many_columns` – слишком много колонок (больше 100);
* `max_missing_share` – максимальная доля пропусков по любой колонке;
* `too_many_missing` – флаг, что доля пропусков в какой-то колонке превышает 50 %;
* `quality_score` – интегральная оценка качества в диапазоне `[0, 1]`.

Новые эвристики:

* `has_constant_columns` – есть ли константные колонки (одно уникальное значение);
* `constant_columns` – список имён таких колонок;
* `has_high_cardinality_categoricals` – есть ли категориальные колонки с высокой кардинальностью;
* `high_cardinality_columns` – список этих колонок.

Флаги используются в `cli.report` и выводятся в разделе «Качество данных (эвристики)» отчёта
`report.md`.

---

## HTTP API (HW04)

Запуск FastAPI через `uvicorn`:

```bash
cd homeworks/HW04/eda-cli
uv sync
uv run uvicorn eda_cli.api:app --reload --port 8000
```

Документация Swagger после старта: `http://127.0.0.1:8000/docs`.

Доступные эндпоинты:

* `GET /health` – проверка работоспособности
* `POST /quality` – оценка по агрегатам (n_rows, n_cols и т.д.)
* `POST /quality-from-csv` – оценка качества по CSV (использует ядро HW03)
* `POST /quality-flags-from-csv` – дополнительные флаги по CSV (использует ядро HW03)

Пример запроса (CSV-файл):

```bash
curl -X POST -F "file=@data/example.csv" http://127.0.0.1:8000/quality-from-csv
```

---

## Тесты

Для проверки корректности функций `core.py` и новых эвристик используется `pytest`.

Запуск из корня проекта `eda-cli`:

```bash
cd homeworks/HW04/eda-cli
uv run pytest -q
```

Тесты проверяют:

* базовую сводку по датасету и таблицу пропусков;
* расчёт корреляции и top-категорий;
* работу новых эвристик `has_constant_columns` и
  `has_high_cardinality_categoricals` в `compute_quality_flags`.

---

## Минимальный сценарий проверки

1. Перейти в проект и установить зависимости:

   ```bash
   cd homeworks/HW04/eda-cli
   uv sync
   ```

2. Проверить работу CLI:

   ```bash
   uv run eda-cli overview data/example.csv
   uv run eda-cli report data/example.csv --out-dir reports_example
   uv run uvicorn eda_cli.api:app --reload --port 8000
   ```

3. Убедиться, что все тесты проходят:

   ```bash
   uv run pytest -q
   ```

Если все команды выполняются без ошибок, в `reports_example/` появляются артефакты отчёта
и API стартует через `uvicorn`, домашнее задание HW04 считается технически выполненным.
