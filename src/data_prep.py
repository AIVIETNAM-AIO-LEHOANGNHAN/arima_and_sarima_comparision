"""
============================================================
src/data_prep.py — Xử lý dữ liệu (Data Preparation)
============================================================
Hàm chính:
    load_and_split_data(path) → (train, test)

Chức năng:
    1. Đọc CSV (date, sales)
    2. Làm sạch: gắn cờ ngày nghỉ (sales=0), đánh dấu outliers
    3. Trích xuất đặc trưng thời gian
    4. Chia Train / Test theo thời gian (time-based split)
============================================================
"""

import numpy as np
import pandas as pd


def load_and_split_data(path: str, split_date: str = "2017-04-01"):
    """
    Đọc dữ liệu từ file CSV, làm sạch, trích xuất features,
    và chia train/test theo thời gian.

    Parameters
    ----------
    path : str
        Đường dẫn đến file CSV (phải có cột 'date' và 'sales').
    split_date : str
        Ngày chia train/test (format: 'YYYY-MM-DD').
        Mặc định: '2017-04-01'.

    Returns
    -------
    train : pd.DataFrame
        Dữ liệu training (trước split_date).
    test : pd.DataFrame
        Dữ liệu testing (từ split_date trở đi).
    """

    # ----------------------------------------------------------
    # 1. Đọc dữ liệu
    # ----------------------------------------------------------
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ----------------------------------------------------------
    # 2. Làm sạch dữ liệu
    # ----------------------------------------------------------
    # 2a. Gắn cờ ngày nghỉ (sales = 0)
    df["is_holiday"] = (df["sales"] == 0).astype(int)

    # 2b. Phát hiện outliers bằng IQR (chỉ trên ngày không nghỉ)
    df_active = df[df["is_holiday"] == 0]["sales"]
    Q1 = df_active.quantile(0.25)
    Q3 = df_active.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_mask = (df["sales"] < lower_bound) | (df["sales"] > upper_bound)
    outliers_mask = outliers_mask & (df["is_holiday"] == 0)
    df["is_outlier"] = outliers_mask.astype(int)

    # ----------------------------------------------------------
    # 3. Trích xuất đặc trưng thời gian (Feature Engineering)
    # ----------------------------------------------------------
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek       # 0=Mon, 6=Sun
    df["day_name"] = df["date"].dt.day_name()
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear

    # Biến sin/cos cho tính chu kỳ
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ----------------------------------------------------------
    # 4. Chia Train / Test theo thời gian
    # ----------------------------------------------------------
    split = pd.Timestamp(split_date)
    train = df[df["date"] < split].copy()
    test = df[df["date"] >= split].copy()

    return train, test
