import numpy as np
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
        errors = []
        X, Y = reduce_to_train_data(data[i:index - threshold], metric)

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

    if not regressors:
        return None

    mse_errors = list(map(lambda x: x['mse'], regressors))
    min_error_index = np.argmin(mse_errors)

    return regressors[min_error_index]


def get_serie_data(raw_data, threshold, base_index=30, metric="cases"):

    new_data = []

    data = raw_data[base_index:]

    for i, row in enumerate(data):
        best_model = get_best_model(
            raw_data, base_index + i, threshold, metric)

        if not best_model:
            continue

        def pred_fn(n):
            return math.floor(best_model['regressor'](n) + 0.5)

        f_actual = best_model['Y'][-1]
        f_prediction = pred_fn(len(best_model['X']) - 1)
        pred_diff = f_actual - f_prediction

        pred_index = len(best_model['X']) + threshold
        pred_value = pred_fn(pred_index) + pred_diff

        raw_value = get_metric(raw_data[base_index + i], metric)

        if pred_value != 0:
            error_from_raw = (pred_value - raw_value) / pred_value
        else:
            error_from_raw = 0

        new_data.append({
            'x': row.date,
            'y': error_from_raw * 100,
            'is_prediction': True,
            'raw_value': raw_value,
            'pred_value': pred_value,
            'raw_error': pred_value - raw_value,
        })

    return new_data[-base_index:]


def get_predictions(data, metric, next_days, test_size=7):
    _, test_Y = reduce_to_train_data(data[-test_size:], metric)

    regressors = []

    for i in range(len(data)):
        X, Y = reduce_to_train_data(data[-(2 * test_size + i):], metric)
        X = X[:test_size + i]
        Y = Y[:test_size + i]

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

    best_model = regressors[min_error_index]

    def pred_fn(n):
        return math.floor(best_model['regressor'](n) + 0.5)

    # Calculate new preds
    new_preds = []
    best_X = best_model['X']

    f_actual = test_Y[-1]
    f_prediction = pred_fn(len(best_X) + test_size - 1)
    pred_diff = f_actual - f_prediction

    last_metric = get_metric(data[-1], metric)

    for i in range(next_days):
        pred_value = pred_fn(len(best_X) + test_size + i) + pred_diff

        if pred_value > last_metric:
            last_metric = pred_value

        new_preds.append({
            'date': f"new-date-{i}",
            f"{metric}": max(pred_value, last_metric)
        })

    return new_preds
