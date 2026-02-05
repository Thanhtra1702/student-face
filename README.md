# Hệ Thống Điểm Danh Sinh Viên Thông Minh (AI Kiosk) - Cầm tay & HD

Dự án Kiosk điểm danh sử dụng công nghệ nhận diện khuôn mặt tiên tiến (ArcFace + Mediapipe), giao diện Web App hiện đại, tối ưu hóa cho độ phân giải HD 720p và trải nghiệm người dùng cao cấp.

## 🚀 Tính Năng Chính

- **Nhận diện khuôn mặt HD:** Xử lý thời gian thực trên khung hình 1280x720, hình ảnh sắc nét, tracking mượt mà.
- **Smart Snapshot:** Khi nhận diện thành công, hệ thống tự động "đóng băng" camera và vẽ khung xanh xác nhận chuyên nghiệp.
- **Clean Snapshot Logic:** Lưu trữ song song bản ảnh có khung (để hiển thị) và bản ảnh SẠCH (để nạp AI), đảm bảo dữ liệu tự học đạt độ chính xác tuyệt đối.
- **Phân loại khoảng cách:** Hệ thống chỉ kích hoạt nhận diện khi sinh viên đứng trong khoảng cách tối ưu (1.5m - 2m).
- **Tốc độ cực nhanh (Fast Path):** Tự động bỏ qua bước xác thực lần 2 nếu độ tin cậy đạt trên 65% (Score > 0.65).
- **Multi-Vector & Augmentation:** Tạo ra 8 biến thể (xoay, sáng, tối, tương phản...) cho mỗi ảnh mẫu để AI nhận diện tốt trong mọi điều kiện ánh sáng.

## 🛠 Yêu Cầu Hệ Thống

- **OS:** Windows 10/11, macOS, hoặc Linux.
- **Python:** 3.8 - 3.10 (Khuyên dùng 3.10).
- **Webcam:** Hỗ trợ HD 720p.
- **Thư viện:** mediapipe, deepface, flask, qdrant-client, opencv-python.

## 📦 Cài Đặt

### 1. Clone dự án

```bash
git clone https://github.com/Thanhtra1702/Face-Recognition.git
cd student_face
```

### 2. Tạo môi trường ảo (Khuyên dùng)

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

## 🗄 Khởi Tạo Dữ Liệu (Lần đầu chạy)

1. **Khởi tạo Qdrant (Vector DB):**

   ```bash
   python init_qdrant.py
   ```

2. **Khởi tạo SQLite (Metadata DB):**

   ```bash
   python setup_database.py
   ```

## 📸 Quy Trình Thêm/Cập Nhật Dữ Liệu

### Tự học từ Kiosk (Smart Self-Learning)

- Khi sinh viên điểm danh và bấm **Xác nhận**, hệ thống tự động lưu **Ảnh SẠCH** (không dính khung vẽ) vào `collected_faces/{MSSV}/`.
- Chạy `python process_collected_faces.py` định kỳ để nạp dữ liệu mới.
- **Lưu ý:** Ảnh sau khi xử lý sẽ được di chuyển vào thư mục `processed/` để đối soát thay vì bị xóa.

## 🖥 Chạy Ứng Dụng Kiosk

```bash
python app.py
```

- Truy cập: `http://localhost:5000`
- Bấm **F11** để vào chế độ Toàn màn hình.

## 🔒 Thông số tối ưu (Current Config)

- **Resolution:** 1280x720 (720p HD).
- **Threshold:** 0.45 (Cân bằng Tốc độ/Chính xác).
- **Gap Check:** 0.02 (Lọc nhập nhằng ID khác).
- **Image Enhance:** CLAHE 3.0 (Cân bằng sáng HD).
- **Fast Path:** 0.65 (Xác nhận tức thì).

## 📂 Cấu Trúc Dự Án

```text
📁 student_face/
├── 📄 app.py                     # Web Server & Core AI (HD Logic)
├── 📄 kiosk_db.py                # Database Handler (Qdrant + SQLite)
├── 📄 setup_database.py          # Script khởi tạo SQLite ban đầu
├── 📄 process_collected_faces.py # Xử lý ảnh tự học & Augmentation
├── 📄 init_qdrant.py             # Khởi tạo Vector DB
├── 📁 templates/                 # UI (HTML/CSS)
├── 📁 collected_faces/           # Ảnh chờ xử lý / processed (đã lưu trữ)
├── 📁 database/                  # Ảnh đại diện gốc
└── 📁 qdrant_db/                 # Trí não AI (Vector Database)
```

## ❓ Troubleshooting

- **Nhận diện chậm:** Kiểm tra độ sáng môi trường. Đứng gần camera hơn sao cho khung tracking chuyển sang màu Cam.
- **Nhận diện sai:** Xóa ảnh cũ trong `database/`, chụp lại ảnh mới sắc nét hơn và chạy lại `init_qdrant.py`.
- **Camera lag:** Giảm độ phân giải trong `app.py` xuống 640x480 nếu CPU quá tải.
