# Итог выполнения HW03 – CLI-утилита для EDA

## Кратко

В HW03 основная работа оформлена как Python-проект `eda-cli` с CLI для EDA,
эвристиками качества данных и тестами. Подробности и команды запуска описаны
в `homeworks/HW03/eda-cli/README.md`.

## Структура

```text
homeworks/
  HW03/
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

Файл: `homeworks/HW03/eda-cli/data/example.csv`

Используется как пример для EDA-команд CLI.

## Как воспроизвести

### 1) Зависимости

Находясь в корне проекта `eda-cli`:

```bash
cd homeworks/HW03/eda-cli
uv sync
```

### 2) Запуск

Примеры CLI-команд:

```bash
cd homeworks/HW03/eda-cli
uv run eda-cli overview data/example.csv
uv run eda-cli report data/example.csv --out-dir reports_example
```

## Методология

* CLI-команды для обзора и полного отчёта по датасету
* эвристики качества данных (пропуски, константные и высококардинальные колонки)
* тестирование ключевых функций через `pytest`

## Результаты

* генерируются CSV-отчёты и графики в указанной папке (`reports_example/`)
* тесты подтверждают корректность расчётов и эвристик

## Вывод

HW03 оформляет EDA как повторяемый CLI-инструмент с тестами, пригодный для
дальнейшего расширения.
