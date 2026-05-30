# Самопроверка проекта

| # | Критерий | Да/Нет | Где смотреть / комментарий |
|---|---|---|---|
| 1 | Сервис запускается по инструкциям из `project/README.md` и работает | ✅ | `README.md`, `src/service/app.py`, endpoint `/health` |
| 2 | Endpoint `/predict` использует реальную обученную модель, а не заглушку | ✅ | `src/models/predict.py`, `artifacts/best_model.joblib` |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/01_eda.ipynb`, `notebooks/02_baselines.ipynb`, `artifacts/model_comparison_test.csv` |
| 4 | Есть baseline и сравнение нескольких моделей по метрикам | ✅ | DummyClassifier + 5 sklearn-моделей в `src/models/train.py` |
| 5 | Код не свален в один ноутбук: есть структура в `src/` | ✅ | `src/data/`, `src/models/`, `src/service/` |
| 6 | Есть Dockerfile или понятный сценарий развёртывания без Docker | ✅ | `Dockerfile`, `.dockerignore`, контейнерный `pytest` |
| 7 | Есть `.env.example` и нет реальных секретов | ✅ | `configs/.env.example`, `SECURITY.md`; проект не использует секреты |
| 8 | Реализованы логи/наблюдаемость и `/health` | ✅ | `src/service/app.py`, Docker `HEALTHCHECK` |
| 9 | В `report.md` обоснован выбор финальной модели | ✅ | разделы 6-9 отчёта |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | локальный запуск, FastAPI, Streamlit UI, Docker |

Итого: **10 / 10**.

Комментарий: проект сделан как компактный sklearn ML-пайплайн с API,
Streamlit-интерфейсом, Docker-сценарием, конфигом, security notes и тестами.
