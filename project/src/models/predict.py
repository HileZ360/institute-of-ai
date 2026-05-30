"""Инференс сохранённой модели оттока."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.loader import FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.joblib"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"


class ChurnModel:
    """Обёртка над сохранённым sklearn pipeline и metadata."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        raw_model_path = Path(
            model_path or os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        # В .env путь относительный; без этого Docker и локальный запуск расходятся.
        self.model_path = (
            raw_model_path
            if raw_model_path.is_absolute()
            else PROJECT_ROOT / raw_model_path
        )
        self.metadata_path = Path(metadata_path or DEFAULT_METADATA_PATH)
        self._model: Any | None = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {self.model_path}. "
                "Сначала выполните `python -m src.models.train` из папки project."
            )
        self._model = joblib.load(self.model_path)
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    @property
    def model_name(self) -> str:
        return str(self.metadata.get("best_model", "sklearn_pipeline"))

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _frame_from_features(self, features: dict[str, Any]) -> pd.DataFrame:
        missing = [column for column in FEATURE_COLUMNS if column not in features]
        if missing:
            raise ValueError("Не переданы признаки: " + ", ".join(missing))
        return pd.DataFrame([{column: features[column] for column in FEATURE_COLUMNS}])

    def predict_proba(self, features: dict[str, Any]) -> float:
        frame = self._frame_from_features(features)
        return float(self.model.predict_proba(frame)[0, 1])

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        probability = self.predict_proba(features)
        prediction = int(probability >= 0.5)
        return {
            "churn_probability": probability,
            "prediction": prediction,
            "risk_level": risk_level(probability),
            "model_name": self.model_name,
        }


def risk_level(probability: float) -> str:
    """Возвращает категорию риска по вероятности оттока."""
    if probability < 0.25:
        return "low"
    if probability < 0.50:
        return "medium"
    return "high"
