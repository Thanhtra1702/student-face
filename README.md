# Hệ Thống Điểm Danh Sinh Viên Thông Minh (AI Kiosk)

Dự án Kiosk điểm danh sử dụng công nghệ nhận diện khuôn mặt tiên tiến (ArcFace + Mediapipe), giao diện Web App hiện đại, tích hợp chống giả mạo (Anti-Spoofing).

## 🚀 Tính Năng Chính

- **Nhận diện khuôn mặt:** Real-time với độ chính xác cao (DeepFace ArcFace + Mediapipe).
- **Chống giả mạo (Anti-Spoofing):**
  - Silent Liveness Detection: Yêu cầu chớp mắt tự nhiên (người dùng không biết).
  - Screen Detection: Phát hiện viền thiết bị và độ lóa màn hình.
- **Xác nhận 3 lần (Triple Verification):** Phải nhận đúng 3 frame liên tiếp mới xác nhận.
- **Tự học (Self-Learning):** Mỗi lần xác nhận, hệ thống tự lưu ảnh để cải thiện độ chính xác.
- **Giao diện Kiosk:** Web-based (HTML/CSS/JS) hiện đại, animation mượt mà.
- **Multi-Vector:** Hỗ trợ lưu nhiều ảnh/vector cho mỗi sinh viên (càng nhiều càng chính xác).

## 🛠 Yêu Cầu Hệ Thống

- **OS:** Windows 10/11, macOS, hoặc Linux.
- **Python:** 3.8 - 3.10 (Khuyên dùng 3.10).
- **Webcam:** Kết nối sẵn sàng.
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
# Linux/Mac:
source venv/bin/activate
```

### 3. Cài đặt thư viện

```bash
pip install -r requirements.txt
pip install mediapipe  # Bắt buộc cho Anti-Spoofing
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

### Cách 1: Thêm thủ công (Batch)

1. Tạo thư mục con theo MSSV trong `collected_faces/`:
   ```
   collected_faces/
   ├── QE170001/
   │   ├── anh1.jpg
   │   ├── anh2.jpg
   │   └── anh3.jpg
   └── QE170002/
       └── photo.png
   ```

2. Chạy xử lý:
   ```bash
   python process_collected_faces.py
   ```

3. Đồng bộ thông tin:
   ```bash
   python sync_database_full.py
   ```

### Cách 2: Tự học từ Kiosk (Self-Learning)

- Khi sinh viên điểm danh và bấm **Xác nhận**, hệ thống tự động lưu ảnh vào `collected_faces/{MSSV}/`.
- Định kỳ (hàng tuần), chạy `python process_collected_faces.py` để nạp dữ liệu mới vào AI.
- Hệ thống càng dùng càng thông minh!

## 🖥 Chạy Ứng Dụng Kiosk

```bash
python app.py
```

- Truy cập: `http://localhost:5000`
- Bấm **F11** để vào chế độ Toàn màn hình.

## 🔒 Cơ Chế Bảo Mật

1. **Liveness Detection:** Hệ thống âm thầm chờ người dùng chớp mắt tự nhiên. Ảnh tĩnh hoặc video không chớp mắt sẽ không vượt qua được.
2. **Screen Detection:** Phát hiện viền điện thoại/tablet và ánh sáng lóa từ màn hình.
3. **Triple Verification:** Phải nhận diện đúng cùng 1 người trong 3 khung hình liên tiếp.

## 📂 Cấu Trúc Dự Án

```
📁 student_face/
├── 📄 app.py                     # Web Server (Flask) & Core AI Logic
├── 📄 kiosk_db.py                # Database Handler (Qdrant + SQLite)
├── 📄 process_collected_faces.py # Tool xử lý ảnh -> Vector (Multi-Vector)
├── 📄 sync_database_full.py      # Tool đồng bộ Metadata
├── 📁 templates/                 # HTML UI
├── 📁 static/                    # CSS, JS, Assets
├── 📁 database/                  # Ảnh đại diện (1 ảnh/người)
├── 📁 collected_faces/           # Ảnh chờ xử lý (theo thư mục MSSV)
└── 📁 qdrant_db/                 # Vector Database (Multi-Vector/người)
```

## ❓ Troubleshooting

- **Lỗi "No module named mediapipe"**: Chạy `pip install mediapipe`.
- **Camera không lên:** Kiểm tra quyền truy cập Camera hoặc đóng ứng dụng khác đang dùng camera.
- **Nhận diện sai:** Thêm nhiều ảnh đa góc vào `collected_faces/{MSSV}/` và chạy lại `process_collected_faces.py`.
- **Không nhận ra người thật:** Hạ ngưỡng `threshold` trong `app.py` (mặc định 0.55).
