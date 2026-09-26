"""
============================================================
src/metrics.py — Hàm đánh giá RMSLE
============================================================
Hàm chính:
    rmsle(y_true, y_pred) → float

RMSLE = Root Mean Squared Logarithmic Error
    = sqrt( mean( (log(1 + y_true) - log(1 + y_pred))^2 ) )

Ưu điểm:
    - Phạt dự đoán thấp hơn thực tế nặng hơn so với dự đoán cao
    - Ít nhạy cảm với outliers so với RMSE
    - Phù hợp cho dữ liệu sales (skewed, luôn >= 0)
============================================================
"""

import numpy as np


def rmsle(y_true, y_pred):
    """
    Tính Root Mean Squared Logarithmic Error (RMSLE).

    Parameters
    ----------
    y_true : array-like
        Giá trị thực tế (ground truth). Phải >= 0.
    y_pred : array-like
        Giá trị dự đoán. Phải >= 0.

    Returns
    -------
    float
        Giá trị RMSLE.

    Notes
    -----
    - Các giá trị âm trong y_pred sẽ được clip về 0 trước khi tính.
    - Công thức: sqrt( mean( (log(1 + y_true) - log(1 + y_pred))^2 ) )
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    # Clip giá trị âm về 0 (phòng trường hợp model dự đoán âm)
    y_pred = np.clip(y_pred, 0, None)

    log_true = np.log1p(y_true)   # log(1 + y_true)
    log_pred = np.log1p(y_pred)   # log(1 + y_pred)

    return np.sqrt(np.mean((log_true - log_pred) ** 2))
