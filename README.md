# Pipeline Dự đoán Bệnh Tim Mạch BRFSS (Phiên bản Demo)

Repository này chứa một pipeline Machine Learning từ đầu đến cuối dự đoán nguy cơ mắc bệnh Tim mạch (CVD) dựa trên dữ liệu khảo sát Hệ thống Giám sát Yếu tố Rủi ro Hành vi (BRFSS) của CDC.

## Các tính năng

- **Xử lý dữ liệu**: Pipeline PySpark để làm sạch, xử lý các giá trị bị thiếu và kỹ thuật tính toán đặc trưng.
- **Huấn luyện mô hình**: Đánh giá nhiều mô hình (LightGBM, Random Forest, SVM) và sử dụng SMOTE để xử lý mất cân bằng lớp.
- **REST API**: Backend FastAPI phục vụ dự đoán từ mô hình được huấn luyện tốt nhất.
- **Dashboard**: Dùng Streamlit để dự đoán tương tác và trực quan hóa dữ liệu.
- **Containerization**: Được Docker hóa để triển khai liền mạch.

## Quick start

Có thể chạy toàn bộ pipeline cục bộ bằng Docker Compose. Nó sẽ tự động sử dụng bộ dữ liệu mẫu đã bao gồm (`data/sample_brfss_data.csv`).

1. Sao chép kho lưu trữ:
   ```bash
   git clone <your-repo-url>
   cd demo_BI
   ```

2. Chạy ứng dụng:
   ```bash
   docker-compose up --build
   ```

3. Truy cập các dịch vụ:
   - **Dashboard Streamlit**: [http://localhost:8501](http://localhost:8501)
   - **Backend FastAPI (Swagger docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Chạy cục bộ (Không có Docker)

Nếu muốn chạy các tập lệnh trực tiếp qua Python:

1. Tạo môi trường ảo và cài đặt các phụ thuộc:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Trên Windows
   # source .venv/bin/activate  # Trên macOS/Linux
   pip install -r requirements.txt
   ```

2. Chạy pipeline:
   ```bash
   python main.py
   ```

## Sử dụng bộ dữ liệu đầy đủ

Nếu bạn muốn huấn luyện mô hình trên bộ dữ liệu BRFSS đầy đủ (~400.000+ hàng):

1. Tải xuống `brfss_data.csv` đầy đủ (hoặc tệp `LLCP2022.XPT`). 
2. Đặt tệp bên trong thư mục `data/`.
3. Pipeline (`main.py` và các tập lệnh PySpark) sẽ tự động phát hiện `brfss_data.csv` và sử dụng nó thay vì bộ dữ liệu mẫu.

## Kiến trúc

1. **`main.py`**: Trình điều phối trung tâm.
2. **`src/data_processing.py`**: Logic PySpark để làm sạch dữ liệu.
3. **`src/api.py`**: FastAPI phục vụ dự đoán mô hình.
4. **`src/dashboard.py`**: Frontend Streamlit.
5. **`src/kafka_consumer.py`**: Tích hợp Kafka để ghi realtime logs.

---

# BRFSS Cardiovascular Disease Predictive Pipeline (Demo Version)

Welcome! This repository contains an end-to-end Machine Learning pipeline that predicts the risk of Cardiovascular Disease (CVD) based on the CDC's Behavioral Risk Factor Surveillance System (BRFSS) survey data.

> **Note for Recruiters:** To make it incredibly easy for you to test and run this project out-of-the-box, this repository includes a small **2,000-row sample dataset**. You can run the entire pipeline instantly without downloading the original massive (1GB+) BRFSS datasets!

## Features

- **Data Processing**: PySpark pipeline for cleaning, handling missing values, and feature engineering.
- **Model Training**: Evaluates multiple models (LightGBM, Random Forest, SVM) and uses SMOTE for handling class imbalances.
- **REST API**: A FastAPI backend that serves predictions from the best trained model.
- **Dashboard**: A Streamlit dashboard for interactive predictions and data visualization.
- **Containerization**: Fully Dockerized for seamless deployment.

## Quick Start (Out-of-the-Box)

You can run the entire pipeline locally using Docker Compose. It will automatically use the included sample dataset (`data/sample_brfss_data.csv`).

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd demo_BI
   ```

2. Run the application:
   ```bash
   docker-compose up --build
   ```

3. Access the services:
   - **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
   - **FastAPI Backend (Swagger Docs)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Running Locally (Without Docker)

If you prefer to run the scripts directly via Python:

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On macOS/Linux
   pip install -r requirements.txt
   ```

2. Run the orchestrator pipeline:
   ```bash
   python main.py
   ```
   *This will run the PySpark data cleaning step, train the ML models, save the artifacts to the `models/` directory, and launch the FastAPI server.*

## Using the Full Dataset

If you want to train the model on the full BRFSS dataset (~400,000+ rows):

1. Download the full `brfss_data.csv` (or the `LLCP2022.XPT` file).
2. Place the file inside the `data/` directory.
3. The pipeline (`main.py` and PySpark scripts) will automatically detect `brfss_data.csv` and use it instead of the sample dataset.

## Architecture

1. **`main.py`**: The central orchestrator.
2. **`src/data_processing.py`**: PySpark logic for data cleaning.
3. **`src/api.py`**: FastAPI serving the model predictions.
4. **`src/dashboard.py`**: Streamlit frontend.
5. **`src/kafka_consumer.py`**: Kafka integration for real-time inference logging.

## Security & Production Notes

⚠️ **This is a demo project for educational and recruitment purposes only. Do NOT use in production without implementing the security measures below.**

### Current Demo Limitations

- **No API Authentication**: Endpoints are publicly accessible. Implement OAuth2 or API key authentication for production.
- **CORS Misconfiguration**: Currently allows requests from any origin (`"*"`). Restrict to specific domains in production.
- **No Encryption**: Kafka communication uses PLAINTEXT protocol. Enable SSL/TLS for encrypted message transport.
- **Hardcoded Service Addresses**: Kafka bootstrap servers are hardcoded. Externalize via environment variables or configuration files.
- **No Input Validation**: Patient data is not strictly validated. Add comprehensive input sanitization and validation.
- **No Rate Limiting**: The API has no rate limiting. Implement rate limiting to prevent abuse.



