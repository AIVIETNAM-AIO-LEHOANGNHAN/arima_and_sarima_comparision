"""
============================================================
src/arima_sarima.py — ARIMA & SARIMA Modeling
============================================================
Hàm chính:
    check_stationarity(series)  → dict  (ADF + KPSS test)
    fit_sarimax(series, order, seasonal_order, trend, maxiter)
                                → SARIMAXResults (đã assert hội tụ)
    check_ljungbox(model, lags) → pd.DataFrame
    rmsle(y_true, y_pred)       → float
============================================================
"""

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


def check_stationarity(series, alpha=0.05):
    """
    Kiểm định tính dừng bằng ADF test VÀ KPSS test.

    Parameters
    ----------
    series : pd.Series
        Chuỗi thời gian cần kiểm định.
    alpha : float, default=0.05
        Mức ý nghĩa.

    Returns
    -------
    dict
        Kết quả gồm ADF và KPSS.
    """
    series_clean = series.dropna()

    # ADF test
    adf_result = adfuller(series_clean, regression="ct")

    # KPSS test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_result = kpss(series_clean, regression="ct", nlags="auto")

    return {
        'adf_stat': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_lags': adf_result[2],
        'adf_nobs': adf_result[3],
        'adf_critical': adf_result[4],
        'adf_stationary': adf_result[1] < alpha,
        'kpss_stat': kpss_result[0],
        'kpss_pvalue': kpss_result[1],
        'kpss_lags': kpss_result[2],
        'kpss_critical': kpss_result[3],
        'kpss_stationary': kpss_result[1] >= alpha,
    }


def fit_sarimax(series, order, seasonal_order=None,
                trend='c', maxiter=500):
    """
    Fit mô hình SARIMAX chính thức, kiểm tra hội tụ.

    Parameters
    ----------
    series : pd.Series
        Chuỗi training.
    order : tuple
        (p, d, q).
    seasonal_order : tuple or None
        (P, D, Q, m). None nếu ARIMA thuần.
    trend : str, default='c'
        Trend component.
    maxiter : int, default=500
        Số vòng lặp tối đa.

    Returns
    -------
    SARIMAXResultsWrapper
        Kết quả đã fit. Assert converged=True.
    """
    if seasonal_order is None:
        seasonal_order = (0, 0, 0, 0)

    model = sm.tsa.statespace.SARIMAX(
        series,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(disp=False, maxiter=maxiter)

    converged = result.mle_retvals.get('converged', False)
    assert converged, (
        f"Model chưa hội tụ: order={order}, "
        f"seasonal={seasonal_order}"
    )

    return result


def check_ljungbox(model, lags=None):
    """
    Kiểm định Ljung-Box, loại đúng loglikelihood_burn.

    Parameters
    ----------
    model : SARIMAXResultsWrapper
        Model đã fit.
    lags : list or None
        Các lag cần kiểm định.
        Default: [7, 14, 21, 28, 35, 42, 49].

    Returns
    -------
    pd.DataFrame
        Kết quả Ljung-Box (lb_stat, lb_pvalue).
    """
    if lags is None:
        lags = [7, 14, 21, 28, 35, 42, 49]

    burn = model.loglikelihood_burn
    resid_trimmed = model.resid.iloc[burn:]

    return acorr_ljungbox(
        resid_trimmed.dropna(),
        lags=lags,
        return_df=True
    )


def rmsle(y_true, y_pred):
    """
    Tính RMSLE (Root Mean Squared Logarithmic Error).

    Parameters
    ----------
    y_true : array-like
        Giá trị thực tế (≥ 0).
    y_pred : array-like
        Giá trị dự đoán.

    Returns
    -------
    float
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean(
        (np.log1p(y_pred) - np.log1p(y_true)) ** 2
    ))
