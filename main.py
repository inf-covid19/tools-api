import time
import concurrent.futures
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
from utils import calculate_prediction_errors
from multiprocessing import Pool, cpu_count

app = FastAPI()


class DailyRecord(BaseModel):
    date: str
    cases: int
    cases_daily: int
    deaths: int
    deaths_daily: int


class PredictionsInput(BaseModel):
    records: List[DailyRecord]
    days: int = 7


class PredictionsErrorInput(BaseModel):
    records: List[DailyRecord]
    thresholds: List[int]
    base_index: int = 30


@app.post("/predictions/{metric}")
def read_item(req: PredictionsInput, metric: str = "cases"):

    records = req.records
    days = req.days

    predictions = calculate_prediction_errors.get_predictions(
        records, metric, days)

    return {
        'predictions': predictions
    }


@app.post("/predictions-error/{metric}")
def read_item(req: PredictionsErrorInput, metric: str = "cases"):

    records = req.records
    thresholds = req.thresholds
    base_index = req.base_index

    result_series = []

    with Pool() as pool:

        values = ((records, t, base_index, metric) for t in thresholds)

        series = pool.starmap(
            calculate_prediction_errors.get_serie_data, values)

        for i, serie in enumerate(series):
            result_series.append({
                'threshold': thresholds[i],
                'serie': serie
            })

    return {
        'series': result_series
    }
