"""Проверки пользовательского интерфейса и состава моделей."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import pytest

from src.models.train import make_models


def _import_ui_module():
    try:
        return importlib.import_module("src.service.ui")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Модуль веб-интерфейса не найден: {exc}")


def test_model_stack_uses_sklearn_estimators_for_all_candidates():
    models = make_models()

    assert "GradientBoostingClassifier" in models
    assert all(
        estimator.__class__.__module__.startswith("sklearn.")
        for estimator in models.values()
    )


def test_logistic_regression_uses_stable_solver():
    model = make_models()["LogisticRegression"]

    assert model.solver == "liblinear"


def test_ui_builds_complete_feature_payload_from_defaults():
    ui = _import_ui_module()

    payload = ui.default_feature_values()

    assert set(payload) == set(ui.FEATURE_LABELS)
    assert payload["complaints"] in {0, 1}
    assert payload["tariff_plan"] in {1, 2}
    assert payload["status"] in {1, 2}


def test_ui_formats_prediction_for_display():
    ui = _import_ui_module()

    result = ui.format_prediction(
        {
            "churn_probability": 0.3764,
            "prediction": 1,
            "risk_level": "medium",
            "model_name": "GradientBoostingClassifier",
        }
    )

    assert result["probability_percent"] == "37.64%"
    assert result["prediction_text"] == "клиент уйдёт"
    assert result["risk_text"] == "средний"
    assert result["model_name"] == "GradientBoostingClassifier"


def test_ui_float_number_input_uses_float_bounds():
    ui = _import_ui_module()

    class FakeColumn:
        def number_input(self, **kwargs):
            assert isinstance(kwargs["value"], float)
            assert isinstance(kwargs["min_value"], float)
            assert isinstance(kwargs["step"], float)
            return kwargs["value"]

    value = ui._number_input(
        FakeColumn(),
        "customer_value",
        72.30,
        key_prefix="test",
        step=0.01,
    )

    assert value == 72.30


def test_ui_script_imports_when_streamlit_runs_file_by_path(tmp_path):
    project_root = importlib.import_module("src.data.loader").PROJECT_ROOT
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    code = (
        "import runpy; "
        f"runpy.run_path({str(project_root / 'src' / 'service' / 'ui.py')!r}, "
        "run_name='streamlit_probe')"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_ui_content_has_onboarding_dashboard_and_complete_scenarios():
    ui = _import_ui_module()
    content = importlib.import_module("src.service.ui_content")

    cards = content.dashboard_cards()
    assert [card["label"] for card in cards] == [
        "Финальная модель",
        "Test F1",
        "Test ROC-AUC",
        "Recall",
    ]
    assert cards[0]["value"] == "GradientBoostingClassifier"

    onboarding = content.onboarding_steps()
    assert [step["title"] for step in onboarding] == [
        "Проверить готовность",
        "Выбрать профиль",
        "Рассчитать риск",
    ]

    scenarios = content.demo_scenarios()
    assert {"stable_customer", "at_risk_customer"} <= set(scenarios)
    assert all(
        set(scenario["features"]) == set(ui.FEATURE_LABELS)
        for scenario in scenarios.values()
    )
    assert all(content.feature_help(feature) for feature in ui.FEATURE_LABELS)


def test_docker_files_keep_container_regression_reproducible():
    project_root = importlib.import_module("src.data.loader").PROJECT_ROOT
    dockerfile = (project_root / "Dockerfile").read_text(encoding="utf-8")
    dockerignore = (project_root / ".dockerignore").read_text(encoding="utf-8")

    assert "COPY tests/ tests/" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert ".venv/" in dockerignore
    assert "aie-course-meta/" in dockerignore
