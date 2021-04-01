from datetime import datetime
from typing import List, Optional
from multiprocessing import Pool

from pydantic import BaseModel
from fastapi import FastAPI
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from tools.core.prediction import get_predictions, get_series_data

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(
    title="COVID-19 Analysis Tools API",
    description="API for https://covid19.ufrgs.dev/tools",
    middleware=middleware,
)


class DailyRecord(BaseModel):
    date: datetime
    cases: int
    cases_daily: int
    deaths: int
    deaths_daily: int


class PredictionsInput(BaseModel):
    records: List[DailyRecord]
    days: int = 7


class DailyPrediction(BaseModel):
    date: datetime
    cases: Optional[int]
    deaths: Optional[int]
    is_prediction: bool = True


class PredictionsOutput(BaseModel):
    predictions: List[DailyPrediction]


class PredictionsErrorInput(BaseModel):
    records: List[DailyRecord]
    thresholds: List[int]
    base_index: int = 30


class ErrorSeriesItem(BaseModel):
    x: datetime
    y: float
    is_prediction: bool
    raw_value: int
    pred_value: int
    raw_error: int


class ErrorSeries(BaseModel):
    threshold: int
    data: List[ErrorSeriesItem]


class PredictionsErrorOutput(BaseModel):
    series: List[ErrorSeries]


@app.get("/_ah/health", tags=["system"])
def health():
    return {"is_healthy": True}


@app.post(
    "/api/v1/predictions/{metric}",
    tags=["predictions"],
    response_model=PredictionsOutput,
)
def fetch_predictions(input: PredictionsInput, metric: str):

    records = input.records
    days = input.days

    predictions = get_predictions(records, metric, days)

    return {"predictions": predictions}


@app.post(
    "/api/v1/predictions/{metric}/errors",
    tags=["predictions"],
    response_model=PredictionsErrorOutput,
)
def fetch_predictions_errors(input: PredictionsErrorInput, metric: str):

    records = input.records
    thresholds = input.thresholds
    base_index = input.base_index

    result_series = []

    with Pool() as pool:

        values = ((records, t, base_index, metric) for t in thresholds)

        series = pool.starmap(get_series_data, values)

        for i, data in enumerate(series):
            result_series.append({"threshold": thresholds[i], "data": data})

    return {"series": result_series}
