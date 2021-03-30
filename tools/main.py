from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from multiprocessing import Pool

from tools.utils.calculate_prediction_errors import get_predictions, get_series_data
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DailyRecord(BaseModel):
    date: str
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
    isPrediction: bool = True


class PredictionsOutput(BaseModel):
    predictions: List[DailyPrediction]


class PredictionsErrorInput(BaseModel):
    records: List[DailyRecord]
    thresholds: List[int]
    base_index: int = 30


class ErrorSeriesItem(BaseModel):
    x: datetime
    y: float
    isPrediction: bool
    rawValue: int
    predValue: int
    rawError: int


class ErrorSeries(BaseModel):
    threshold: int
    data: List[ErrorSeriesItem]


class PredictionsErrorOutput(BaseModel):
    series: List[ErrorSeries]


@app.get("/_ah/health", tags=["system"])
def health():
    return {"is_healthy": True}


@app.post("/api/v1/predictions/{metric}", tags=["predictions"], response_model=PredictionsOutput)
def fetch_predictions(req: PredictionsInput, metric: str):

    records = req.records
    days = req.days

    predictions = get_predictions(records, metric, days)

    return {
        'predictions': predictions
    }


@app.post("/api/v1/predictions/{metric}/errors", tags=["predictions"], response_model=PredictionsErrorOutput)
def fetch_predictions_errors(req: PredictionsErrorInput, metric: str):

    records = req.records
    thresholds = req.thresholds
    base_index = req.base_index

    result_series = []

    with Pool() as pool:

        values = ((records, t, base_index, metric) for t in thresholds)

        series = pool.starmap(get_series_data, values)

        for i, data in enumerate(series):
            result_series.append({
                'threshold': thresholds[i],
                'data': data
            })

    return {
        'series': result_series
    }
