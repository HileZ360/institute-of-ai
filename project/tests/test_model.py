"""Минимальные быстрые тесты, чтобы не сломать проект перед сдачей."""

from pathlib import Path

from fastapi.testclient import TestClient

from src.data.loader import (
    DATASET_CANDIDATES,
    DATASET_PATH,
    TARGET_COLUMN,
    load_churn_data,
)
from src.models.predict import ChurnModel, risk_level
from src.service import app as app_module

app = app_module.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.joblib"

SAMPLE_PAYLOAD = {
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
    "customer_value": 197.64,
}


def test_dataset_loads_and_duplicates_are_removed():
    df = load_churn_data()
    assert df.shape == (2850, 14)
    assert TARGET_COLUMN in df.columns
    assert df.duplicated().sum() == 0
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}


def test_project_dataset_is_the_only_default_source():
    assert DATASET_CANDIDATES == [DATASET_PATH]
    assert DATASET_PATH.name == "Customer Churn.csv"
    assert DATASET_PATH.parent.name == "data"


def test_risk_level_thresholds():
    assert risk_level(0.10) == "low"
    assert risk_level(0.30) == "medium"
    assert risk_level(0.70) == "high"


def test_saved_model_predicts_probability():
    assert MODEL_PATH.exists(), "Перед тестами выполните `python -m src.models.train`."
    model = ChurnModel(MODEL_PATH)
    result = model.predict(SAMPLE_PAYLOAD)
    assert 0 <= result["churn_probability"] <= 1
    assert result["prediction"] in {0, 1}
    assert result["risk_level"] in {"low", "medium", "high"}


def test_api_health_and_predict():
    assert MODEL_PATH.exists(), "Перед тестами выполните `python -m src.models.train`."
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["model_loaded"] is True

        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert 0 <= data["churn_probability"] <= 1
        assert data["prediction"] in {0, 1}
        assert data["risk_level"] in {"low", "medium", "high"}


def test_api_hides_internal_prediction_errors(monkeypatch):
    class BrokenModel:
        model_name = "broken"

        def predict(self, features):
            raise RuntimeError("internal path /tmp/private/model.joblib leaked")

    with TestClient(app) as client:
        monkeypatch.setattr(app_module, "model", BrokenModel())

        response = client.post("/predict", json=SAMPLE_PAYLOAD)

    assert response.status_code == 400
    assert response.json()["detail"] == "Не удалось выполнить предсказание"
