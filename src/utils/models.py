from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def evaluate_metrics(actual, pred):
    '''Requires from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error'''
    '''Requires two arrays as positional arguments'''
    '''Yields regression metrics based on the real values and the predicion'''
    yield f'rmse: {mean_squared_error(actual, pred, squared=False)}'
    yield f'mse: {mean_squared_error(actual, pred)}'
    yield f'mae: {mean_absolute_error(actual, pred)}'
    yield f'r2 score: {r2_score(actual, pred)}'
    yield f'mape: {mean_absolute_percentage_error(actual, pred)}'
