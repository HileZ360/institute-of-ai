from __future__ import annotations

from typing import Any, Dict, Tuple
from time import perf_counter
from io import BytesIO

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

from eda_cli.core import summarize_dataset, missing_table, compute_quality_flags

APP_VERSION = "0.2.0"

app = FastAPI(
    title="Dataset quality service",
    description="HTTP-сервис эвристической оценки качества табличных датасетов.",
    version=APP_VERSION,
)

class DatasetShape(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)

class QualityRequest(BaseModel):
    n_rows: int = Field(..., ge=0)
    n_cols: int = Field(..., ge=0)
    max_missing_share: float = Field(..., ge=0.0, le=1.0)
    numeric_cols: int = Field(..., ge=0)
    categorical_cols: int = Field(..., ge=0)

class QualityResponse(BaseModel):
    ok_for_model: bool
    quality_score: float = Field(..., ge=0.0, le=1.0)
    message: str
    latency_ms: float
    flags: Dict[str, bool]
    dataset_shape: DatasetShape

class QualityFlagsResponse(BaseModel):
    flags: Dict[str, bool]

def _compute_quality_from_aggregates(req: QualityRequest) -> tuple[Dict[str, bool], float, str]:
    flags: Dict[str, bool] = {}

    flags["too_few_rows"] = req.n_rows < 100
    flags["too_many_columns"] = req.n_cols > 300
    flags["too_many_missing"] = req.max_missing_share > 0.3
    flags["no_numeric_columns"] = req.numeric_cols == 0
    flags["no_categorical_columns"] = req.categorical_cols == 0

    quality_score = 1.0
    if flags["too_few_rows"]:
        quality_score -= 0.3
    if flags["too_many_columns"]:
        quality_score -= 0.2
    if flags["too_many_missing"]:
        quality_score -= 0.3
    if flags["no_numeric_columns"]:
        quality_score -= 0.1
    if flags["no_categorical_columns"]:
        quality_score -= 0.1
    quality_score = max(0.0, min(1.0, quality_score))
    if quality_score >= 0.7 and not flags["too_few_rows"] and not flags["too_many_missing"]:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
        ok_for_model = True
    elif quality_score >= 0.4:
        message = "Данные умеренного качества, модель можно обучать с осторожностью."
        ok_for_model = True
    else:
        message = "Качество данных низкое, модель обучать рискованно."
        ok_for_model = False
    flags["ok_for_model"] = ok_for_model
    return flags, quality_score, message


def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Файл не передан.")
    try:
        content = file.file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Файл пустой.")
        df = pd.read_csv(BytesIO(content))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV не содержит строк.")
    return df

def _normalize_quality_output(quality: Any) -> tuple[Dict[str, Any], float]:
    if isinstance(quality, tuple) and len(quality) == 2:
        raw, score = quality
        return dict(raw), float(score)
    if isinstance(quality, dict):
        raw = quality.get("flags", quality)
        score = quality.get("quality_score", 1.0)
        return dict(raw), float(score)
    raw = getattr(quality, "flags")
    score = getattr(quality, "quality_score", 1.0)
    return dict(raw), float(score)


def _to_bool_flags(raw: Dict[str, Any]) -> Dict[str, bool]:
    flags: Dict[str, bool] = {}

    for k, v in raw.items():
        if isinstance(v, bool):
            flags[k] = v
    if "constant_columns" in raw:
        try:
            flags["has_constant_columns"] = len(raw["constant_columns"]) > 0
        except Exception:
            flags["has_constant_columns"] = bool(raw["constant_columns"])
    if "high_cardinality_columns" in raw:
        try:
            flags["has_high_cardinality_columns"] = len(raw["high_cardinality_columns"]) > 0
        except Exception:
            flags["has_high_cardinality_columns"] = bool(raw["high_cardinality_columns"])
    if "max_missing_share" in raw and "too_many_missing" not in flags:
        try:
            flags["too_many_missing"] = float(raw["max_missing_share"]) > 0.3
        except Exception:
            flags["too_many_missing"] = False
    flags.pop("quality_score", None)
    return flags


def _compute_quality_from_df(df: pd.DataFrame) -> tuple[Dict[str, bool], float]:
    summary = summarize_dataset(df)
    missing = missing_table(df)

    quality = compute_quality_flags(summary, missing)
    raw, quality_score = _normalize_quality_output(quality)
    flags = _to_bool_flags(raw)
    quality_score = max(0.0, min(1.0, float(quality_score)))
    return flags, quality_score

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": APP_VERSION,
    }

@app.post("/quality", response_model=QualityResponse)
def quality(request: QualityRequest) -> QualityResponse:
    start = perf_counter()
    flags, quality_score, message = _compute_quality_from_aggregates(request)
    latency_ms = (perf_counter() - start) * 1000.0

    return QualityResponse(
        ok_for_model=flags.pop("ok_for_model", quality_score >= 0.5),
        quality_score=quality_score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape=DatasetShape(n_rows=request.n_rows, n_cols=request.n_cols),
    )

@app.post("/quality-from-csv", response_model=QualityResponse)
def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    start = perf_counter()
    df = _read_csv_upload(file)
    flags, quality_score = _compute_quality_from_df(df)
    latency_ms = (perf_counter() - start) * 1000.0

    ok_for_model = (
        not flags.get("too_few_rows", False)
        and not flags.get("too_many_missing", False)
        and quality_score >= 0.5
    )

    message = (
        "Данных достаточно, модель можно обучать (по эвристикам из HW03)."
        if ok_for_model
        else "По эвристикам из HW03 качество данных недостаточно для надёжной модели."
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=quality_score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape=DatasetShape(n_rows=df.shape[0], n_cols=df.shape[1]),
    )
@app.post("/quality-flags-from-csv", response_model=QualityFlagsResponse)
def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    df = _read_csv_upload(file)
    flags, _ = _compute_quality_from_df(df)
    return QualityFlagsResponse(flags=flags)
