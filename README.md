# [Bài tập nhóm phần ứng dụng Machine Learning]

## Thông tin nhóm 04

Các thành viên:

-   Mai Cường Thịnh: 20120196
-   Nguyễn Hữu Thiện: 20120194
-   Ngô Nguyễn Quang Tú: 20120234
-   Lưu Tuấn Khanh: 1712522

Xây dựng trang DashBoard phân tích trading theo các tiêu chí sau

1. Người dùng chọn một trong các phương pháp dự đoán :
   a. XGBoost, RNN, LSTM (bắt buộc) hoặc các phương pháp khác.
   b. Transformer and Time Embeddings(nâng cao - có thể làm hoặc không, có điểm cộng)
2. Người dùng chọn một hay nhiều đặc trưng để dự đoán :
   a. Close, ROC (bắt buộc)
   b. RSI, Bolling Bands, Moving Average,...(nâng cao)
   c. Đường hỗ trợ/kháng cự (nâng cao)
3. Hiển thị giá dự đoán của timeframe kế tiếp (visualize)
4. Lấy dữ liệu Real-time từ websocket của Binance, chứng khoán,...
   a. Lấy dữ liệu lớn hơn 1000 nến(candle) từ lịch sử
   b. Lấy dữ liệu real-time append vô dữ liệu hiện tại
   c. Dự đoán giá nến(candle) kế tiếp
   Xây dựng DashBoard theo tutorial sau : **[Stock Price Prediction – Machine Learning Project in Python](https://data-flair.training/blogs/stock-price-prediction-machine-learning-project-in-python/)**
   Tiêu chí chấm : Hỗ trợ càng nhiều mô hình dự đoán(Kết hợp phương pháp và các đặc trưng) càng tốt, giao diện đẹp, tiện dụng, dễ xài.

## Cách cài đặt dự án

Bước 1: Tải packages pip install -r requirements.txt

Bước 2: Tạo folder **"data"** bên trong folder dự án để lưu trữ các dữ liệu về tiền tệ.

Bước 3:

-   Bước 3.1:

Build lại thông qua câu lệnh **"python model.py"**

-   Bước 3.2:

Chạy project thông qua câu lệnh **"python app.py"**

Bước 4: Dự án được deploy lên đường dẫn **"htpp://localhost:8050"** trên browser

## Video demo dự án

Sẽ được cập nhật sau
