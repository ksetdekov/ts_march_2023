import numpy as np
import pandas as pd
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression


def extract_trend(ts: pd.Series):
    """Извлекает линейный тренд из временного ряда

    Args:
        ts (pd.Series): временной ряд

    Returns:
        (pd.Seris, float, float): ряд прогноза тренда и коэффициенты
    """
    indices_fake = np.indices([np.array(ts).shape[0]])
    reg = LinearRegression().fit(indices_fake.reshape(-1, 1),
                                 np.array(ts).reshape(-1, 1))
    trend_pred = reg.predict(indices_fake.reshape(-1, 1))

    trend = pd.Series(trend_pred.reshape((-1)), index=ts.index)
    k = reg.coef_
    b = reg.intercept_
    return trend, k, b


def extract_seasonality(ts_detrended: pd.Series, period: int = None):
    """Извлекает сезонную компоненту

    Args:
        ts_detrended (pd.Series): входящий временной ряд
        period (int, optional): Длина периода, если не задавать - оценка через periodogram. Defaults to None.

    Returns:
        pd.Series: временной ряд сезонности
    """
    season = ts_detrended
    # если нет - оценим
    if period is None:
        f, ppx = periodogram(ts_detrended)
        top_feq = list(sorted(zip(f, ppx), key=lambda x: x[1]))[-1]
        period = np.round(1/top_feq[0], 0)

    # detrend
    season = ts_detrended - ts_detrended.diff(6)
    return season


def detect_ts(ts: pd.Series) -> tuple:
    """Runs stl decompositions

    Args:
        ts (pd.Series): input time series

    Returns:
        tuple: (tuple of trend, seasonality, resids)
    """
    trend = extract_trend(ts)[0]
    ts_detrended = ts - trend
    season = extract_seasonality(ts_detrended)
    resid = ts_detrended - season
    return (trend, season, resid)
