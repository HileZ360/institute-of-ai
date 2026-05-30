# Итоговый проект: прогнозирование оттока абонентов телеком-оператора

Учебный проект решает задачу бинарной классификации для телеком-оператора:
по обезличенным табличным признакам абонента нужно оценить вероятность оттока
(`Churn = 1`) и вернуть категорию риска. В качестве источника данных
используется датасет Iranian Churn.

Реализация закрывает требования итогового проекта курса: есть EDA, сравнение
моделей, сохранённый sklearn pipeline, FastAPI-сервис с `/health` и `/predict`,
локальный веб-интерфейс, Dockerfile, конфиги, логи, security notes и тесты.

## Стек

- Python 3.11+.
- ML и анализ данных: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.
- Сервис: `FastAPI`, `uvicorn`, `pydantic`.
- Веб-интерфейс для демонстрации: `Streamlit`.
- Тесты: `pytest`, `httpx`.

## Структура

```text
project/
├── README.md
├── report.md
├── self-checklist.md
├── SECURITY.md
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── configs/.env.example
├── data/Customer Churn.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baselines.ipynb
├── src/
│   ├── data/loader.py
│   ├── models/train.py
│   ├── models/predict.py
│   └── service/
│       ├── app.py
│       ├── ui_content.py
│       └── ui.py
├── artifacts/
│   ├── best_model.joblib
│   ├── model_metadata.json
│   ├── model_comparison_cv.csv
│   ├── model_comparison_test.csv
│   └── figures/
└── tests/
    ├── test_model.py
    └── test_ui_and_model_stack.py
```

## Данные

- Используемый датасет: `data/Customer Churn.csv`.
- Исходный размер: 3150 строк, 14 колонок.
- Exact-дубликаты: 300 строк.
- После удаления дубликатов: 2850 строк, 13 признаков и target.
- Target: `Churn`; в коде после нормализации колонок используется `churn`.
- Баланс классов после очистки: `0` — 2404 (84.35%), `1` — 446 (15.65%).

Дубликаты удаляются до train/test split, чтобы одинаковые записи не оказались
одновременно в обучении и тесте.

## Установка

Все команды ниже выполняются из папки `project/`. Если терминал уже показывает
путь вида `.../institute-of-ai/project`, команду `cd project` выполнять не
нужно. Если вы находитесь в корне репозитория `institute-of-ai`, сначала
выполните:

```bash
cd project
```

После этого создайте окружение и установите зависимости:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Для Windows PowerShell активация окружения отличается:

```powershell
.\.venv\Scripts\Activate.ps1
```

Если используется `uv`, можно запускать команды без ручной установки:

```bash
uv run --no-project --with-requirements requirements.txt python -m pytest tests -q
```

## Обучение и тесты

Пересоздать модель, таблицы метрик и графики:

```bash
python -m src.models.train
```

Запустить тесты:

```bash
python -m pytest tests -q
```

## API

Запуск:

```bash
python -m src.service.app
```

Сервис доступен на `http://localhost:8000`. Swagger UI доступен на
`http://localhost:8000/docs`.

Health-check:

```bash
curl http://localhost:8000/health
```

Пример предсказания:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "call_failures": 8,
    "complaints": 0,
    "subscription_length": 38,
    "charge_amount": 0,
    "seconds_of_use": 4370,
    "frequency_of_use": 71,
    "frequency_of_sms": 5,
    "distinct_called_numbers": 17,
    "age_group": 3,
    "tariff_plan": 1,
    "status": 1,
    "age": 30,
    "customer_value": 197.64
  }'
```

Ответ содержит вероятность оттока, бинарный прогноз, категорию риска и имя
модели.

## Веб-интерфейс

Streamlit-интерфейс — GUI-версия учебного проекта. Он использует тот же
`artifacts/best_model.joblib`, что и API.
Первый экран показывает карточки качества модели, демо-профили для быстрой
проверки и форму со справками по каждому признаку.

```bash
streamlit run src/service/ui.py
```

В форме доступны все 13 признаков клиента. После расчёта интерфейс показывает
вероятность оттока, прогноз, уровень риска, имя модели и короткую интерпретацию
для демонстрации.

## Модели и результат

Сравниваются:

- `DummyClassifier(strategy="most_frequent")`;
- `LogisticRegression`;
- `DecisionTreeClassifier`;
- `RandomForestClassifier`;
- `GradientBoostingClassifier`;
- `KNeighborsClassifier`.

Протокол: `train_test_split(test_size=0.2, stratify=y, random_state=42)` и
`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

Лучшая модель по test F1: `GradientBoostingClassifier`.

| Модель | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| GradientBoostingClassifier | 0.9526 | 0.9306 | 0.7528 | 0.8323 | 0.9818 |
| KNeighborsClassifier | 0.9351 | 0.8333 | 0.7303 | 0.7784 | 0.9686 |
| RandomForestClassifier | 0.8789 | 0.5694 | 0.9213 | 0.7039 | 0.9690 |
| DecisionTreeClassifier | 0.8596 | 0.5306 | 0.8764 | 0.6610 | 0.9219 |
| LogisticRegression | 0.8368 | 0.4877 | 0.8876 | 0.6295 | 0.9288 |
| DummyClassifier | 0.8439 | 0.0000 | 0.0000 | 0.0000 | 0.5000 |

Выбор модели подробно обоснован в `report.md`.

## Docker

Docker-сценарий запускает FastAPI-сервис. Модель должна быть обучена до сборки
образа, чтобы `artifacts/best_model.joblib` попал внутрь контейнера.

```bash
python -m src.models.train
docker build -t iranian-churn-classic-ml .
docker run --rm iranian-churn-classic-ml python -m pytest tests -q
docker run -p 8000:8000 --env-file configs/.env.example iranian-churn-classic-ml
```

После запуска проверить:

```bash
curl http://localhost:8000/health
```

Лог контейнерного регрессионного прогона можно сохранить так:

```bash
mkdir -p artifacts/test-runs
docker run --rm iranian-churn-classic-ml python -m pytest tests -q \
  | tee artifacts/test-runs/docker-pytest.txt
```

## Безопасность

- В проекте нет реальных секретов; используется только `configs/.env.example`.
- Вход `/predict` валидируется Pydantic-схемой FastAPI.
- Ошибки инференса возвращаются клиенту нейтрально, подробности остаются в логах.
- Подробный handoff перед публикацией описан в `SECURITY.md`.

## Демонстрация на защите

1. Показать структуру `project/`, `self-checklist.md` и `report.md`.
2. Открыть `notebooks/01_eda.ipynb`: качество данных, дубликаты, баланс классов.
3. Открыть `notebooks/02_baselines.ipynb`: сравнение моделей и метрики.
4. Запустить `python -m src.service.app` и проверить `/health`, `/predict`, `/docs`.
5. Запустить `streamlit run src/service/ui.py` и показать предсказание через форму.
6. Запустить контейнерный regression test: `docker run --rm iranian-churn-classic-ml python -m pytest tests -q`.
7. Кратко объяснить, почему выбран `GradientBoostingClassifier`.
