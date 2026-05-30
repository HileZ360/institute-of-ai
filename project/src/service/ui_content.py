"""Тексты и демо-профили Streamlit-интерфейса."""

from __future__ import annotations

from typing import Any


MODEL_NAME = "GradientBoostingClassifier"

DASHBOARD_CARDS = [
    {
        "label": "Финальная модель",
        "value": MODEL_NAME,
        "caption": "выбрана по test F1, затем ROC-AUC и Recall",
    },
    {
        "label": "Test F1",
        "value": "0.8323",
        "caption": "баланс precision и recall для класса оттока",
    },
    {
        "label": "Test ROC-AUC",
        "value": "0.9818",
        "caption": "качество ранжирования клиентов по риску",
    },
    {
        "label": "Recall",
        "value": "0.7528",
        "caption": "найдено около 75% ушедших клиентов на test",
    },
]

ONBOARDING_STEPS = [
    {
        "title": "Проверить готовность",
        "body": "Health API должен показывать загруженную модель, а карточки выше фиксируют качество финального pipeline.",
    },
    {
        "title": "Выбрать профиль",
        "body": "Два демо-профиля дают преподавателю быстрый контроль низкого и высокого риска без ручного подбора всех полей.",
    },
    {
        "title": "Рассчитать риск",
        "body": "Результат показывает вероятность, бинарный прогноз, уровень риска и модель, которая дала ответ.",
    },
]

FEATURE_HELP = {
    "call_failures": "Количество неуспешных звонков. Больше сбоев обычно повышает риск ухода.",
    "complaints": "Наличие жалобы: 0 - нет, 1 - есть.",
    "subscription_length": "Сколько месяцев клиент пользуется услугами оператора.",
    "charge_amount": "Категория начислений/тарифа из исходного датасета.",
    "seconds_of_use": "Суммарная длительность звонков. Низкая активность часто связана с churn.",
    "frequency_of_use": "Число звонков за период наблюдения.",
    "frequency_of_sms": "Число SMS за период наблюдения.",
    "distinct_called_numbers": "Сколько разных номеров набирал клиент.",
    "age_group": "Возрастная группа из датасета, значения 1-5.",
    "tariff_plan": "Тарифный план: 1 или 2.",
    "status": "Статус клиента в исходных данных: 1 или 2. Для production нужен аудит момента фиксации.",
    "age": "Возраст клиента.",
    "customer_value": "Расчётная ценность клиента для оператора.",
}

DEMO_SCENARIOS: dict[str, dict[str, Any]] = {
    "stable_customer": {
        "label": "Стабильный клиент",
        "summary": "Активный абонент без жалоб и с высокой регулярностью использования.",
        "features": {
            "call_failures": 2,
            "complaints": 0,
            "subscription_length": 40,
            "charge_amount": 1,
            "seconds_of_use": 7200,
            "frequency_of_use": 115,
            "frequency_of_sms": 80,
            "distinct_called_numbers": 35,
            "age_group": 3,
            "tariff_plan": 1,
            "status": 1,
            "age": 32,
            "customer_value": 420.50,
        },
    },
    "at_risk_customer": {
        "label": "Клиент в зоне риска",
        "summary": "Есть жалоба, ниже активность и слабее клиентская ценность.",
        "features": {
            "call_failures": 12,
            "complaints": 1,
            "subscription_length": 12,
            "charge_amount": 0,
            "seconds_of_use": 980,
            "frequency_of_use": 18,
            "frequency_of_sms": 2,
            "distinct_called_numbers": 6,
            "age_group": 2,
            "tariff_plan": 1,
            "status": 2,
            "age": 27,
            "customer_value": 72.30,
        },
    },
}

RESULT_GUIDANCE = {
    "low": "Клиент выглядит устойчивым. Достаточно обычного мониторинга.",
    "medium": "Есть сигналы риска. Лучше проверить жалобы и активность клиента.",
    "high": "Высокий риск ухода. Это кандидат на удерживающее действие.",
}


def dashboard_cards() -> list[dict[str, str]]:
    return [dict(card) for card in DASHBOARD_CARDS]


def onboarding_steps() -> list[dict[str, str]]:
    return [dict(step) for step in ONBOARDING_STEPS]


def demo_scenarios() -> dict[str, dict[str, Any]]:
    return {
        key: {
            "label": value["label"],
            "summary": value["summary"],
            "features": dict(value["features"]),
        }
        for key, value in DEMO_SCENARIOS.items()
    }


def feature_help(feature: str) -> str:
    return FEATURE_HELP[feature]


def result_guidance(risk_level: str) -> str:
    return RESULT_GUIDANCE.get(risk_level, "Проверьте входные признаки и бизнес-контекст клиента.")
