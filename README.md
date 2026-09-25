# So sánh hiệu quả dự báo doanh thu bán lẻ giữa ARIMA và SARIMA

So sánh khả năng dự báo doanh số bán lẻ ngắn hạn giữa mô hình **ARIMA** và **SARIMA**, sử dụng dữ liệu thật từ **Corporación Favorita** (Kaggle – *Store Sales: Time Series Forecasting*). Mô hình tốt nhất được kiểm định bằng kiểm định thống kê chính thức.

---

## 📌 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Câu hỏi & Giả thuyết nghiên cứu](#-câu-hỏi--giả-thuyết-nghiên-cứu)
- [Dữ liệu](#-dữ-liệu)
- [Quy trình thực hiện](#-quy-trình-thực-hiện)
- [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
- [Cài đặt & Chạy dự án](#-cài-đặt--chạy-dự-án)
- [Kết quả chính](#-kết-quả-chính)
- [Quy tắc cộng tác](#-quy-tắc-cộng-tác)
- [Thành viên nhóm](#-thành-viên-nhóm)

---

## 🎯 Giới thiệu

Dự án sử dụng dữ liệu doanh số bán hàng thật của **Corporación Favorita** — tập đoàn bán lẻ thực phẩm lớn tại Ecuador — được công bố trong cuộc thi Kaggle *Store Sales – Time Series Forecasting*.

**Mục tiêu:** đánh giá và so sánh hiệu quả của hai mô hình chuỗi thời gian đơn biến (ARIMA và SARIMA) trong việc dự báo doanh số ngắn hạn, làm cơ sở hỗ trợ ra quyết định vận hành (quản lý tồn kho, lập kế hoạch nhân sự) cho một cửa hàng bán lẻ.

## ❓ Câu hỏi & Giả thuyết nghiên cứu

**RQ1 (So sánh):** Giữa ARIMA và SARIMA, mô hình nào cho sai số dự báo (RMSLE) thấp hơn khi dự báo doanh số cửa hàng trong năm 2017?

**RQ2 (Tính khả dụng):** Mô hình tốt nhất có cải thiện đáng kể so với mô hình Baseline (Seasonal Naive) hay không, đủ để coi là có giá trị ứng dụng thực tế?

**H1:** SARIMA cho RMSLE thấp hơn có ý nghĩa so với ARIMA thuần, do dữ liệu đã xác nhận có tính mùa vụ.

**H2:** Cả ARIMA và SARIMA đều vượt trội hơn Baseline.

## 🗂 Dữ liệu

| Thuộc tính | Thông tin |
|---|---|
| Nguồn | [Kaggle – Store Sales: Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting) |
| Dữ liệu gốc | 3.000.888 quan sát, 54 cửa hàng, 1/1/2013 – 15/8/2017 |
| Dữ liệu nghiên cứu | 1 cửa hàng (Store 1), tổng hợp theo ngày → 1.688 quan sát |
| Train | 2013-01-01 → 2016-12-31 |
| Test | 2017-01-01 → 2017-08-15 |
| Metric đánh giá | RMSLE (Root Mean Squared Logarithmic Error) — theo đúng chuẩn cuộc thi |

## 🔄 Quy trình thực hiện

```
Task 1: Baseline           →  Mốc so sánh khách quan + làm sạch dữ liệu, chia Train/Test
Task 2: Decomposition       →  Phân rã Trend – Seasonal (tuần/tháng/quý/năm) – Residual
Task 3: Kiểm định mùa vụ    →  QS test xác nhận mùa vụ nào thực sự tồn tại (không đoán mò)
Task 4: ARIMA & SARIMA      →  Xây, chẩn đoán (Ljung-Box), dự báo, tính RMSLE
Task 5: Kết luận            →  Trả lời RQ1/RQ2, kiểm định Diebold-Mariano, viết báo cáo
```

Mỗi Task là một mắt xích — Task sau luôn sử dụng dữ liệu/kết quả đã được Task trước xử lý và lưu lại

**Nguyên tắc bắt buộc:** mọi bước phân tích, kiểm định, xác định tham số (Task 2, 3, và fit model ở Task 4) **chỉ được thực hiện trên tập Train**. Tập Test (2017) chỉ được dùng duy nhất ở bước dự báo và đánh giá cuối cùng, nhằm tránh rò rỉ dữ liệu (data leakage).

## 📁 Cấu trúc thư mục

```
store-sales-forecasting/
├── data/
│   ├── raw/                     # Dữ liệu gốc từ Kaggle (train.csv, holidays_events.csv, stores.csv)
│   └── processed/                # Dữ liệu đã xử lý, dùng chung cho cả nhóm
├── notebooks/                    # 1 notebook / task
│   ├── 01_baseline.ipynb
│   ├── 02_decomposition.ipynb
│   ├── 03_seasonality_tests.ipynb
│   ├── 04_arima_sarima.ipynb
│   └── 05_conclusion.ipynb
├── src/                           # Hàm dùng chung (import vào notebook)
│   ├── data_prep.py
│   ├── metrics.py
│   ├── baseline.py
│   ├── decompose.py
│   └── seasonality_tests.py
├── results/
│   ├── figures/                   # Biểu đồ (.png)
│   ├── metrics/                   # Kết quả số liệu (.csv)
│   └── models/                    # Model đã fit (.pkl)
├── reports/
│   └── final_report.md            # Báo cáo tổng hợp cuối cùng
├── docs/
│   └── PROJECT_DESCRIPTION.md     # Mô tả chi tiết từng task
├── requirements.txt
└── README.md
```

## ⚙️ Cài đặt & Chạy dự án

```bash
# Clone repo
git clone https://github.com/<tên-nhóm>/store-sales-forecasting.git
cd store-sales-forecasting

# Tạo môi trường ảo (khuyến nghị)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

**Thư viện chính sử dụng:** `pandas`, `numpy`, `statsmodels`, `pmdarima`, `scipy`, `matplotlib`

**Dữ liệu:** tải bộ dữ liệu từ [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), giải nén vào `data/raw/`.

**Chạy tuần tự:** mở và chạy lần lượt các notebook trong `notebooks/` theo đúng thứ tự 01 → 05 (mỗi notebook phụ thuộc vào output của notebook trước, được lưu trong `data/processed/` và `results/`).

## 📈 Kết quả chính

| Model | RMSLE |
|---|---|
| Baseline (Seasonal Naive) | *cập nhật sau khi hoàn thành Task 1* |
| ARIMA | *cập nhật sau khi hoàn thành Task 4* |
| SARIMA | *cập nhật sau khi hoàn thành Task 4* |

Xem chi tiết đầy đủ (bảng so sánh, kiểm định Diebold-Mariano, phân tích sai số, giới hạn nghiên cứu) tại [`reports/final_report.md`](reports/final_report.md).

## 🤝 Quy tắc cộng tác

- Mỗi Task làm trên **1 branch riêng**: `feature/taskN-<tên-task>`, hoàn thành thì tạo **Pull Request** vào `main`, không push thẳng.
- Dữ liệu đã xử lý trong `data/processed/` là nguồn dùng chung duy nhất — đọc từ đây, không tự tạo lại từ `raw/`.
- Hàm dùng nhiều lần viết trong `src/`, import vào notebook — không copy-paste code qua lại.
- Không đưa dữ liệu Test (2017) vào bất kỳ bước phân tích/kiểm định tham số nào (chỉ dùng ở bước dự báo cuối).

Chi tiết yêu cầu, hướng dẫn thực hiện và tiêu chí hoàn thành của từng Task được quản lý trên board Jira/Kanban của nhóm (KAN-2 → KAN-8).

## 👥 Thành viên nhóm

| Họ tên | Vai trò |
|---|---|
| *Nguyễn Quốc Bảo* | *Task 1 – Baseline* |
| *Trần Bá Thục* | *Task 2 – Decomposition* |
| *Trần Bá Thục* | *Task 3 – Kiểm định mùa vụ* |
| *Nguyễn Quốc Bảo* | *Task 4 – ARIMA & SARIMA* |
| *Lê Hoàng Nhân* | *Task 5 – Kết luận & Đề xuất* |
| *Nguyễn Thị Trân Châu* | *Task 6 – Làm slide* |
---

<p align="center"><i>Trường Đại học Tài chính - Marketing</i></p>
