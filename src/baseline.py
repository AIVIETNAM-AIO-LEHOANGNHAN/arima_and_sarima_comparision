"""
============================================================
src/baseline.py — Seasonal Naive Forecast (Baseline Model)
============================================================
Hàm chính:
    seasonal_naive_forecast(train, n_forecast, m=7, use_avg_weeks=4)
        → pd.Series (dự đoán)

Seasonal Naive (Average):
    Dự đoán ngày t = trung bình doanh thu cùng ngày trong tuần
    của use_avg_weeks tuần gần nhất.

    Ví dụ (m=7, use_avg_weeks=4):
        Dự đoán thứ Hai tuần tới = (sales thứ Hai tuần này
        + sales thứ Hai tuần trước + ... ) / 4
============================================================
"""

import numpy as np
import pandas as pd


def seasonal_naive_forecast(train, n_forecast, m=7, use_avg_weeks=4):
    """
    Tạo dự đoán Seasonal Naive dựa trên trung bình nhiều tuần.

    Parameters
    ----------
    train : pd.Series
        Chuỗi doanh thu từ tập train (đã sort theo thời gian).
    n_forecast : int
        Số ngày cần dự đoán (= len(test)).
    m : int, default=7
        Chu kỳ mùa vụ (7 = tuần).
    use_avg_weeks : int, default=4
        Số tuần gần nhất dùng để tính trung bình.

    Returns
    -------
    np.ndarray
        Mảng giá trị dự đoán có độ dài n_forecast.

    Notes
    -----
    - Với m=7, use_avg_weeks=4:
        Dự đoán thứ Hai tuần tới = trung bình sales của 4 thứ Hai
        gần nhất trong tập train.
    - Nếu không đủ use_avg_weeks tuần lịch sử cho một ngày,
      sẽ dùng số tuần có sẵn.

    Example
    -------
    >>> pred = seasonal_naive_forecast(train['sales'], len(test), m=7, use_avg_weeks=4)
    """
    train_vals = train.values
    predictions = np.zeros(n_forecast)

    # Kết hợp train values để tham chiếu ngược
    history = list(train_vals)

    for i in range(n_forecast):
        # Lấy giá trị cùng ngày trong tuần từ use_avg_weeks tuần gần nhất
        # Vị trí: history[-m], history[-2*m], ..., history[-use_avg_weeks*m]
        values = []
        for w in range(1, use_avg_weeks + 1):
            idx = len(history) - w * m
            if idx >= 0:
                values.append(history[idx])

        # Tính trung bình (nếu không có giá trị nào, fallback = 0)
        if values:
            predictions[i] = np.mean(values)
        else:
            predictions[i] = 0.0

        # Thêm dự đoán vào lịch sử để dùng cho các bước tiếp theo
        history.append(predictions[i])

    return predictions
