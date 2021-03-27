from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI

from utils import calculate_prediction_errors

app = FastAPI()


class DateRecord(BaseModel):
    date: str
    cases: int
    cases_daily: int
    deaths: int
    deaths_daily: int


@app.post("/predictions/{metric}")
def read_item(records: List[DateRecord], metric: str = "cases"):

    predictions = calculate_prediction_errors.get_predictions(records, metric)

    return {
        'preds': predictions
    }


@app.post("/predictions-error/{threshold}/{metric}")
def read_item(records: List[DateRecord], threshold: int, metric: str = "cases"):

    serie_1d = calculate_prediction_errors.get_serie_data(
        records, threshold, metric=metric)

    return {
        'serie_1d': serie_1d
    }
