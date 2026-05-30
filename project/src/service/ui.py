"""Веб-интерфейс для демонстрации модели оттока."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import ChurnModel
from src.service.ui_content import (
    dashboard_cards,
    demo_scenarios,
    feature_help,
    onboarding_steps,
    result_guidance,
)

FEATURE_LABELS = {
    "call_failures": "Неудачные вызовы",
    "complaints": "Жалобы",
    "subscription_length": "Длительность подписки, мес.",
    "charge_amount": "Уровень начислений",
    "seconds_of_use": "Секунды разговоров",
    "frequency_of_use": "Частота звонков",
    "frequency_of_sms": "Количество SMS",
    "distinct_called_numbers": "Уникальные номера",
    "age_group": "Возрастная группа",
    "tariff_plan": "Тарифный план",
    "status": "Статус клиента",
    "age": "Возраст",
    "customer_value": "Ценность клиента",
}

DEFAULT_FEATURES = {
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

RISK_LABELS = {
    "low": "низкий",
    "medium": "средний",
    "high": "высокий",
}

PREDICTION_LABELS = {
    0: "клиент останется",
    1: "клиент уйдёт",
}


def default_feature_values() -> dict[str, int | float]:
    """Возвращает стартовый пример клиента для формы предсказания."""
    return dict(DEFAULT_FEATURES)


def format_prediction(result: dict[str, Any]) -> dict[str, str]:
    """Готовит ответ модели к отображению в интерфейсе."""
    probability = float(result["churn_probability"])
    prediction = int(result["prediction"])
    risk_level = str(result["risk_level"])
    return {
        "probability_percent": f"{probability:.2%}",
        "prediction_text": PREDICTION_LABELS[prediction],
        "risk_text": RISK_LABELS.get(risk_level, risk_level),
        "risk_level": risk_level,
        "model_name": str(result["model_name"]),
    }


def _load_model() -> ChurnModel:
    model = ChurnModel()
    model.load()
    return model


def _number_input(
    column: Any,
    key: str,
    value: int | float,
    *,
    key_prefix: str,
    step: int | float = 1,
) -> int | float:
    min_value: int | float = 0 if key != "age_group" else 1
    if isinstance(value, float) or isinstance(step, float):
        min_value = float(min_value)

    kwargs = {
        "label": FEATURE_LABELS[key],
        "key": f"{key_prefix}_{key}",
        "value": value,
        "min_value": min_value,
        "step": step,
        "help": feature_help(key),
    }
    if key == "age":
        kwargs["max_value"] = 120
    if key == "age_group":
        kwargs["max_value"] = 5
    return column.number_input(**kwargs)


def _render_dashboard(st: Any, model: ChurnModel) -> None:
    columns = st.columns(4)
    for column, card in zip(columns, dashboard_cards()):
        column.metric(card["label"], card["value"])
        column.caption(card["caption"])

    with st.expander("Паспорт модели", expanded=False):
        st.write(
            {
                "loaded_model": model.model_name,
                "selection_rule": model.metadata.get("selection_rule"),
                "dataset": model.metadata.get("dataset", {}).get("dataset_path"),
                "test_metrics": model.metadata.get("best_test_metrics"),
            }
        )


def _render_onboarding(st: Any) -> None:
    with st.sidebar:
        st.header("Контрольный сценарий")
        for index, step in enumerate(onboarding_steps(), start=1):
            st.markdown(f"**{index}. {step['title']}**")
            st.caption(step["body"])


def _scenario_choice(st: Any) -> tuple[str, dict[str, Any]]:
    scenarios = demo_scenarios()
    scenario_key = st.sidebar.radio(
        "Демо-профиль",
        options=list(scenarios),
        format_func=lambda key: scenarios[key]["label"],
        help="Профили нужны для быстрой демонстрации разных уровней риска.",
    )
    st.sidebar.caption(scenarios[scenario_key]["summary"])
    return scenario_key, scenarios[scenario_key]


def _render_inputs(
    st: Any,
    defaults: dict[str, int | float],
    *,
    key_prefix: str,
) -> dict[str, int | float]:
    left, middle, right = st.columns(3)
    payload = dict(defaults)

    with left:
        payload["call_failures"] = int(
            _number_input(
                left,
                "call_failures",
                defaults["call_failures"],
                key_prefix=key_prefix,
            )
        )
        payload["complaints"] = int(
            left.selectbox(
                FEATURE_LABELS["complaints"],
                options=[0, 1],
                format_func=lambda value: "нет" if value == 0 else "есть",
                index=int(defaults["complaints"]),
                key=f"{key_prefix}_complaints",
                help=feature_help("complaints"),
            )
        )
        payload["subscription_length"] = int(
            _number_input(
                left,
                "subscription_length",
                defaults["subscription_length"],
                key_prefix=key_prefix,
            )
        )
        payload["charge_amount"] = int(
            _number_input(
                left,
                "charge_amount",
                defaults["charge_amount"],
                key_prefix=key_prefix,
            )
        )
        payload["seconds_of_use"] = int(
            _number_input(
                left,
                "seconds_of_use",
                defaults["seconds_of_use"],
                key_prefix=key_prefix,
            )
        )

    with middle:
        payload["frequency_of_use"] = int(
            _number_input(
                middle,
                "frequency_of_use",
                defaults["frequency_of_use"],
                key_prefix=key_prefix,
            )
        )
        payload["frequency_of_sms"] = int(
            _number_input(
                middle,
                "frequency_of_sms",
                defaults["frequency_of_sms"],
                key_prefix=key_prefix,
            )
        )
        payload["distinct_called_numbers"] = int(
            _number_input(
                middle,
                "distinct_called_numbers",
                defaults["distinct_called_numbers"],
                key_prefix=key_prefix,
            )
        )
        payload["age_group"] = int(
            _number_input(
                middle,
                "age_group",
                defaults["age_group"],
                key_prefix=key_prefix,
            )
        )

    with right:
        payload["tariff_plan"] = int(
            right.selectbox(
                FEATURE_LABELS["tariff_plan"],
                options=[1, 2],
                index=int(defaults["tariff_plan"]) - 1,
                key=f"{key_prefix}_tariff_plan",
                help=feature_help("tariff_plan"),
            )
        )
        payload["status"] = int(
            right.selectbox(
                FEATURE_LABELS["status"],
                options=[1, 2],
                index=int(defaults["status"]) - 1,
                key=f"{key_prefix}_status",
                help=feature_help("status"),
            )
        )
        payload["age"] = int(
            _number_input(right, "age", defaults["age"], key_prefix=key_prefix)
        )
        payload["customer_value"] = float(
            _number_input(
                right,
                "customer_value",
                defaults["customer_value"],
                key_prefix=key_prefix,
                step=0.01,
            )
        )

    return payload


def _render_result(st: Any, result: dict[str, str]) -> None:
    first, second, third, fourth = st.columns(4)
    first.metric("Вероятность оттока", result["probability_percent"])
    second.metric("Прогноз", result["prediction_text"])
    third.metric("Риск", result["risk_text"])
    fourth.metric("Модель", result["model_name"])
    st.info(result_guidance(result["risk_level"]))


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Прогнозирование оттока абонентов телеком-оператора",
        page_icon=None,
        layout="wide",
    )
    st.title("Прогнозирование оттока абонентов телеком-оператора")
    st.caption("Учебный проект, GUI-версия.")

    try:
        model = _load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    _render_onboarding(st)
    scenario_key, scenario = _scenario_choice(st)
    _render_dashboard(st, model)

    tab_predict, tab_notes = st.tabs(["Прогноз", "Ограничения"])

    with tab_predict:
        payload = _render_inputs(st, scenario["features"], key_prefix=scenario_key)

        if st.button("Рассчитать риск", type="primary"):
            result = format_prediction(model.predict(payload))
            _render_result(st, result)

    with tab_notes:
        st.subheader("Что важно сказать при защите")
        st.markdown(
            """
            - Модель выбрана по test F1, потому что задача несбалансирована.
            - `status` оставлен как учебный признак, но в production нужен аудит момента его фиксации.
            - Порог 0.5 подходит для демонстрации; в бизнесе его надо выбирать по стоимости ошибок.
            """
        )


if __name__ == "__main__":
    main()
