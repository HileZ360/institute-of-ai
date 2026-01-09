# Итог выполнения HW04 – CLI + HTTP API для EDA

## Кратко

В HW04 проект `eda-cli` расширен HTTP API на FastAPI, сохранив CLI и эвристики
качества из HW03. Подробности и команды запуска описаны в
`homeworks/HW04/eda-cli/README.md`.

## Структура

```text
homeworks/
  HW04/
    README.md
    eda-cli/
      pyproject.toml
      README.md
      data/
        example.csv
      src/
        eda_cli/
      tests/
```

## Данные

Файл: `homeworks/HW04/eda-cli/data/example.csv`

Используется как пример для CLI и API.

## Как воспроизвести

### 1) Зависимости

Находясь в корне проекта `eda-cli`:

```bash
cd homeworks/HW04/eda-cli
uv sync
```

### 2) Запуск

CLI:

```bash
cd homeworks/HW04/eda-cli
uv run eda-cli overview data/example.csv
uv run eda-cli report data/example.csv --out-dir reports_example
```

API:

```bash
cd homeworks/HW04/eda-cli
uv run uvicorn eda_cli.api:app --reload --port 8000
```

## Методология

* CLI-команды для EDA и генерации отчётов
* эвристики качества данных, повторяемые между CLI и API
* HTTP API на FastAPI для загрузки CSV и получения метрик качества

## Результаты

* артефакты отчёта в `reports_example/`
* API доступно через Swagger (`/docs`) и принимает CSV-файлы

## Вывод

HW04 превращает CLI-инструмент в сервис, расширяя EDA сценарии на HTTP-уровень.
