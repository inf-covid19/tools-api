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


@app.post("/predictions/{metric}")
def read_item(req: PredictionsInput, metric: str = "cases"):

    records = req.records
    days = req.days

    predictions = calculate_prediction_errors.get_predictions(
        records, metric, days)

    return {
        'preds': predictions
    }


@app.post("/predictions-error/{metric}")
def read_item(req: PredictionsErrorInput, metric: str = "cases"):

    start = time.ctime()

    records = req.records
    thresholds = req.thresholds

    result_series = []

    for threshold in thresholds:
        serie = calculate_prediction_errors.get_serie_data(
            records, threshold, metric=metric)

        result_series.append({
            'threshold': threshold,
            'serie': serie
        })

    print(f"Request started at {start}")
    print(f"Request finished at {time.ctime()}")

    return {
        'series': result_series
    }


@app.post("/predictions-error-thread/{metric}")
def read_item(req: PredictionsErrorInput, metric: str = "cases"):

    start = time.ctime()

    records = req.records
    thresholds = req.thresholds

    result_series = []

    # with concurrent.futures.ThreadPoolExecutor() as executor:
    with Pool() as pool:
        # futures = [executor.submit(calculate_prediction_errors.get_serie_data, records, t, metric=metric)
        #            for t in thresholds]

        values = ((records, t, 30, metric) for t in thresholds)

        # print("values", values)
        # print("next", next(values))

        res = pool.starmap(calculate_prediction_errors.get_serie_data, values)
        print(res)

        # print("threads")
        # print(futures)
        for i, f in enumerate(res):
            # serie = f.result()
            serie = f
            result_series.append({
                'threshold': thresholds[i],
                'serie': serie
            })

    print(f"Request started at {start}")
    print(f"Request finished at {time.ctime()}")

    return {
        'series': result_series
    }
