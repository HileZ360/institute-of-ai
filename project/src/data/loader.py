"""Загрузка и базовая проверка датасета Iranian Churn."""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_COLUMN = "churn"
RANDOM_STATE = 42

RAW_TO_CLEAN_COLUMNS = {
    "Call Failure": "call_failures",
    "Complains": "complaints",
    "Subscription Length": "subscription_length",
    "Charge Amount": "charge_amount",
    "Seconds of Use": "seconds_of_use",
    "Frequency of use": "frequency_of_use",
    "Frequency of SMS": "frequency_of_sms",
    "Distinct Called Numbers": "distinct_called_numbers",
    "Age Group": "age_group",
    "Tariff Plan": "tariff_plan",
    "Status": "status",
    "Age": "age",
    "Customer Value": "customer_value",
    "Churn": TARGET_COLUMN,
}

FEATURE_COLUMNS = [
    "call_failures",
    "complaints",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "age",
    "customer_value",
]

CATEGORICAL_FEATURES = ["complaints", "tariff_plan", "status", "age_group"]
NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]

FEATURE_DESCRIPTIONS = {
    "call_failures": "активность и качество связи: число неудачных вызовов",
    "complaints": "жалобы клиента: 1 если жалобы были, иначе 0",
    "subscription_length": "длительность подписки в месяцах",
    "charge_amount": "уровень начислений/стоимости тарифа",
    "seconds_of_use": "активность клиента: суммарные секунды разговоров",
    "frequency_of_use": "активность клиента: число звонков",
    "frequency_of_sms": "активность клиента: число SMS",
    "distinct_called_numbers": "активность клиента: число уникальных контактов",
    "age_group": "возрастная группа клиента",
    "tariff_plan": "тарифный план",
    "status": "статус клиента в системе оператора",
    "age": "возраст клиента",
    "customer_value": "расчётная ценность клиента для оператора",
}

DATASET_PATH = PROJECT_ROOT / "data" / "Customer Churn.csv"
DATASET_CANDIDATES = [DATASET_PATH]


def normalize_raw_column(name: str) -> str:
    """В CSV местами двойные пробелы в заголовках."""
    return re.sub(r"\s+", " ", str(name).strip())


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит исходные названия колонок к стабильному snake_case."""
    renamed = {}
    for column in df.columns:
        normalized = normalize_raw_column(column)
        renamed[column] = RAW_TO_CLEAN_COLUMNS.get(
            normalized,
            normalized.lower().replace(" ", "_"),
        )
    return df.rename(columns=renamed)


def display_path(path: str | Path) -> str:
    """Для отчётов лучше без абсолютного пути и имени пользователя."""
    return os.path.relpath(Path(path).resolve(), PROJECT_ROOT)


def find_dataset(candidates: Iterable[Path] | None = None) -> Path:
    """Возвращает путь к основному датасету проекта."""
    checked = []
    for candidate in candidates or DATASET_CANDIDATES:
        path = Path(candidate)
        checked.append(path)
        if path.exists():
            return path

    checked_text = "\n".join(f"- {display_path(path)}" for path in checked)
    raise FileNotFoundError(
        "Датасет Iranian Churn не найден. Проверенные пути:\n" + checked_text
    )


def _read_dataset_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            csv_members = [
                name for name in archive.namelist() if name.lower().endswith(".csv")
            ]
            if not csv_members:
                raise FileNotFoundError(f"В архиве нет CSV-файлов: {display_path(path)}")
            with archive.open(csv_members[0]) as file:
                return pd.read_csv(file)
    return pd.read_csv(path)


def load_churn_data(
    path: str | Path | None = None,
    *,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Загружает датасет и удаляет полные дубликаты до разбиения.

    Это защищает test set от попадания строк, идентичных обучающим примерам.
    """
    dataset_path = Path(path) if path is not None else find_dataset()
    df = clean_columns(_read_dataset_file(dataset_path))

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(
            "В датасете нет обязательных колонок: " + ", ".join(missing)
        )

    df = df[required_columns].copy()
    if remove_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df


def dataset_quality_summary(path: str | Path | None = None) -> dict:
    """Возвращает сводку качества данных для отчёта и metadata."""
    dataset_path = Path(path) if path is not None else find_dataset()
    raw = clean_columns(_read_dataset_file(dataset_path))
    clean = raw[FEATURE_COLUMNS + [TARGET_COLUMN]].drop_duplicates()

    return {
        "dataset_path": display_path(dataset_path),
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "duplicates": int(raw.duplicated().sum()),
        "missing_values": int(raw.isna().sum().sum()),
        "clean_rows": int(clean.shape[0]),
        "clean_columns": int(clean.shape[1]),
        "target_counts": {
            str(k): int(v)
            for k, v in clean[TARGET_COLUMN].value_counts().sort_index().items()
        },
        "target_share": {
            str(k): round(float(v), 4)
            for k, v in clean[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index()
            .items()
        },
    }


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Разделяет таблицу на признаки и целевую переменную."""
    return df[FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()
