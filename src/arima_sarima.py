"""
============================================================
src/arima_sarima.py — ARIMA & SARIMA Modeling
============================================================
Hàm chính:
    check_stationarity(series)  → dict  (ADF test result)
    fit_arima(train, order)     → ARIMA model result
    fit_sarima(train, order, seasonal_order)
                                → SARIMAX model result
    ljung_box_test(residuals, lags)
                                → pd.DataFrame
    forecast_and_evaluate(model_result, n_forecast, y_true, model_name)
                                → dict  (predictions, RMSLE, ...)
============================================================
"""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox


def check_stationarity(series, alpha=0.05):
    """
    Kiểm định tính dừng (Stationarity) bằng ADF test.

    Parameters
    ----------
    series : pd.Series
        Chuỗi thời gian cần kiểm định.
    alpha : float, default=0.05
        Mức ý nghĩa.

    Returns
    -------
    dict
        Kết quả ADF test gồm: adf_stat, p_value, n_lags,
        n_obs, critical_values, is_stationary.
    """
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'adf_stat': result[0],
        'p_value': result[1],
        'n_lags': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < alpha
    }


def fit_arima(train_series, order=(1, 1, 1)):
    """
    Fit mô hình ARIMA.

    Parameters
    ----------
    train_series : pd.Series
        Chuỗi training với DatetimeIndex.
    order : tuple, default=(1,1,1)
        Bậc (p, d, q) của ARIMA.

    Returns
    -------
    ARIMAResultsWrapper
        Kết quả fit model.
    """
    model = ARIMA(train_series, order=order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit()
    return result


def fit_sarima(train_series, order=(1, 1, 1),
               seasonal_order=(1, 1, 1, 7)):
    """
    Fit mô hình SARIMA (SARIMAX không có exog).

    Parameters
    ----------
    train_series : pd.Series
        Chuỗi training với DatetimeIndex.
    order : tuple, default=(1,1,1)
        Bậc (p, d, q) của phần non-seasonal.
    seasonal_order : tuple, default=(1,1,1,7)
        Bậc (P, D, Q, m) của phần seasonal.

    Returns
    -------
    SARIMAXResultsWrapper
        Kết quả fit model.
    """
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp=False, maxiter=500)
    return result


def ljung_box_test(residuals, lags=10):
    """
    Kiểm định Ljung-Box trên phần dư.

    Parameters
    ----------
    residuals : array-like
        Phần dư của mô hình.
    lags : int, default=10
        Số lag dùng trong kiểm định.

    Returns
    -------
    pd.DataFrame
        Kết quả kiểm định Ljung-Box (lb_stat, lb_pvalue).
    """
    result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    return result


def forecast_and_evaluate(model_result, n_forecast, y_true,
                          model_name='Model'):
    """
    Dự báo và tính RMSLE.

    Parameters
    ----------
    model_result : statsmodels result
        Kết quả fit model (ARIMA hoặc SARIMA).
    n_forecast : int
        Số bước dự báo.
    y_true : array-like
        Giá trị thực tế để so sánh.
    model_name : str
        Tên mô hình (dùng cho display).

    Returns
    -------
    dict
        predictions: np.ndarray, rmsle: float, model_name: str
    """
    from src.metrics import rmsle as calc_rmsle

    forecast = model_result.forecast(steps=n_forecast)
    predictions = np.clip(forecast.values, 0, None)

    score = calc_rmsle(y_true, predictions)

    return {
        'model_name': model_name,
        'predictions': predictions,
        'rmsle': score
    }
