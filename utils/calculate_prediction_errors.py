from numpy.core.numeric import identity
from numpy.lib.index_tricks import _fill_diagonal_dispatcher
import pandas as pd
import numpy as np
import json
import math

input_data = {}
output = {}


def get_metric(obj, metric):
    if metric == "deaths":
        return obj.deaths
    return obj.cases


def reduce_to_train_data(data, metric):
    X, Y = [], []

    for i, d in enumerate(data):
        value = get_metric(d, metric)
        X.append(i)
        Y.append(value)

    return X, Y


def get_best_model(data, index, threshold, metric):
    _, test_Y = reduce_to_train_data(data[index-threshold:index], metric)

    regressors = []

    for i in range(index):
        X, Y = reduce_to_train_data(data[i:index - threshold], metric)
        errors = []

        try:
            for v in [2]:
                degree = v if len(X) > 2 else 1

                regressor = np.poly1d(np.polyfit(X, Y, degree))

                for idx, real_value in enumerate(test_Y):
                    pred_value = math.floor(regressor(len(Y) + idx) + 0.5)

                    errors.append(math.pow(real_value - pred_value, 2))

            regressors.append({
                'regressor': regressor,
                'mse': np.mean(errors),
                'X': X,
                'Y': Y
            })
        except:
            pass

    mse_errors = list(map(lambda x: x['mse'], regressors))
    min_error_index = np.argmin(mse_errors)

    return regressors[min_error_index]


def get_serie_data(raw_data, threshold, base_index=30, metric="cases"):

    new_data = []

    data = raw_data[base_index:]

    for i, row in enumerate(data):
        # print(
        #     f"Trying search serie data with {i}/{len(data)} and threshold = {threshold}")
        best_model = get_best_model(
            raw_data, base_index + i, threshold, metric)

        def pred_fn(n):
            return math.floor(best_model['regressor'](n) + 0.5)

        f_actual = best_model['Y'][-1]
        f_prediction = pred_fn(len(best_model['X']) - 1)
        pred_diff = f_actual - f_prediction

        pred_index = len(best_model['X']) + threshold
        pred_value = pred_fn(pred_index) + pred_diff

        raw_value = get_metric(raw_data[base_index + i], metric)
        error_from_raw = (pred_value - raw_value) / pred_value

        new_data.append({
            'x': row.date,
            'y': error_from_raw * 100,
            'is_prediction': True,
            'raw_value': raw_value,
            'pred_value': pred_value,
            'raw_error': pred_value - raw_value,
        })

    return new_data[-base_index:]
