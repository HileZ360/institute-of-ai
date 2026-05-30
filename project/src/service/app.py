"""API для демонстрации модели на защите."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.predict import ChurnModel

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger(__name__)

model: ChurnModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = ChurnModel()
    model.load()
    LOGGER.info("Модель загружена: %s", model.model_name)
    yield
    LOGGER.info("Остановка сервиса")


app = FastAPI(
    title="Прогнозирование оттока абонентов телеком-оператора API",
    description="Учебный проект для прогноза оттока абонента по сохранённой sklearn-модели.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChurnRequest(BaseModel):
    call_failures: int = Field(..., ge=0)
    complaints: int = Field(..., ge=0, le=1)
    subscription_length: int = Field(..., ge=0)
    charge_amount: int = Field(..., ge=0)
    seconds_of_use: int = Field(..., ge=0)
    frequency_of_use: int = Field(..., ge=0)
    frequency_of_sms: int = Field(..., ge=0)
    distinct_called_numbers: int = Field(..., ge=0)
    age_group: int = Field(..., ge=1, le=5)
    tariff_plan: int = Field(..., ge=1, le=2)
    status: int = Field(..., ge=1, le=2)
    age: int = Field(..., ge=0, le=120)
    customer_value: float = Field(..., ge=0)


class ChurnResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    prediction: int = Field(..., ge=0, le=1)
    risk_level: str
    model_name: str


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": model is not None and model.is_loaded,
        "model_name": model.model_name if model is not None else None,
    }


@app.post("/predict", response_model=ChurnResponse)
async def predict(request: ChurnRequest) -> ChurnResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    start = time.perf_counter()
    try:
        result = model.predict(request.model_dump())
    except Exception as exc:
        LOGGER.exception("Ошибка при предсказании")
        raise HTTPException(
            status_code=400,
            detail="Не удалось выполнить предсказание",
        ) from exc

    elapsed = time.perf_counter() - start
    LOGGER.info(
        "prediction=%s probability=%.4f risk=%s elapsed=%.4fs",
        result["prediction"],
        result["churn_probability"],
        result["risk_level"],
        elapsed,
    )
    return ChurnResponse(**result)


if __name__ == "__main__":
    uvicorn.run(
        "src.service.app:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=False,
    )
