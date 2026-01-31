# Hệ Thống Điểm Danh Sinh Viên Thông Minh (AI Kiosk)

Dự án Kiosk điểm danh sử dụng công nghệ nhận diện khuôn mặt tiên tiến (ArcFace), giao diện Web App hiện đại, hỗ trợ người dùng tương tác trực quan.

## 🚀 Tính Năng Chính

- **Nhận diện khuôn mặt:** Real-time với độ chính xác cao bằng DeepFace (ArcFace engine).
- **Giao diện Kiosk:** Web-based (HTML/CSS/JS) hiện đại, animation mượt mà, hỗ trợ màn hình cảm ứng.
- **Workflow Tự đông:** Tự động crop khuôn mặt, trích xuất vector và cập nhật cơ sở dữ liệu.
- **Xử lý thông minh:** Tự động phát hiện và cảnh báo đeo khẩu trang/kính râm (future upgrade), hiển thị trạng thái xử lý trực quan.

## 🛠 Yêu Cầu Hệ Thống

- **OS:** Windows 10/11, macOS, hoặc Linux.
- **Python:** 3.8 - 3.10 (Khuyên dùng 3.10).
- **Webcam:** Kết nối sẵn sàng.

## 📦 Cài Đặt

### 1. Clone dự án

```bash
git clone https://github.com/your-repo/student-face-kiosk.git
cd student-face-kiosk
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
```

*Lưu ý: Nếu gặp lỗi cài đặt `dlib` hoặc `deepface`, hãy đảm bảo bạn đã cài `CMake` và `C++ Build Tools`.*

## 🗄 Khởi Tạo Dữ Liệu (Lần đầu chạy)

Nếu đây là lần đầu bạn chạy dự án, hãy khởi tạo cơ sở dữ liệu:

1. **Khởi tạo Qdrant (Vector DB):**

   ```bash
   python init_qdrant.py
   ```

   *Script này sẽ tạo thư mục `qdrant_db` để lưu trữ vector khuôn mặt.*

2. **Khởi tạo SQLite (Metadata DB):**

   ```bash
   python setup_database.py
   ```

   *Tạo file `student_info.db` chứa thông tin sinh viên (Tên, MSSV, Lịch học).*

## 📸 Quy Trình Thêm/Cập Nhật Dữ Liệu Mới

Hệ thống hỗ trợ quy trình "Learning" tự động từ ảnh chụp hoặc ảnh thẻ.

### Bước 1: Thu thập ảnh

- Copy ảnh của sinh viên vào thư mục `collected_faces/`.
- **Quy tắc đặt tên file:** `MSSV_TenBietDanh.jpg` (Ví dụ: `QE170123_Tuan.jpg` hoặc chỉ cần `QE170123.jpg`).
- Hệ thống hỗ trợ định dạng `.jpg`, `.png`, `.jpeg`.

### Bước 2: Chạy xử lý "Batch Processor"

Chạy lệnh sau để hệ thống tự động:

1. Tìm khuôn mặt trong ảnh.
2. Cắt (Crop) khuôn mặt chuẩn và lưu vào `database/`.
3. Trích xuất Vector và nạp vào Qdrant.
4. Di chuyển ảnh gốc vào thư mục `collected_faces/processed/` (lưu trữ).

```bash
python process_collected_faces.py
```

### Bước 3: Đồng bộ thông tin (Metadata)

Sau khi có ảnh trong database, cần cập nhật thông tin (Tên, Lịch học) vào SQLite:

```bash
python sync_database_full.py
```

*Lưu ý: Mặc định script này sẽ tạo tên là "Sinh viên [MSSV]". Bạn nên dùng DB Browser for SQLite để sửa lại tên và lịch học cho chính xác, hoặc chỉnh sửa script `crawl_fap.py` để lấy dữ liệu thật.*

## 🖥 Chạy Ứng Dụng Kiosk

Sau khi data đã sẵn sàng, hãy khởi chạy Kiosk:

```bash
python app.py
```

- Mở trình duyệt và truy cập: `http://localhost:5000` (hoặc `http://127.0.0.1:5000`).
- Bấm **F11** để vào chế độ Toàn màn hình (Kiosk Mode).

## 📂 Cấu Trúc Dự Án

```
📁 student_face/
├── 📄 app.py                     # Web Server (Flask) & Core Logic
├── 📄 kiosk_db.py                # Database Handler (Qdrant + SQLite)
├── 📄 process_collected_faces.py # Tool xử lý ảnh thô -> Vector
├── 📄 sync_database_full.py      # Tool đồng bộ Metadata
├── 📁 templates/                 # HTML UI
├── 📁 static/                    # CSS, JS, Logo
├── 📁 database/                  # Ảnh khuôn mặt đã crop (Clean Data)
├── 📁 collected_faces/           # Nơi bỏ ảnh mới vào để train
└── 📁 qdrant_db/                 # Vector Database Storage
```

## ❓ Troubleshooting

- **Lỗi "No module named..."**: Kiểm tra lại `pip install -r requirements.txt`.
- **Lỗi Camera không lên:** Kiểm tra quyền truy cập Camera của trình duyệt hoặc ứng dụng khác đang chiếm camera.
- **Nhận diện sai:** Thêm nhiều ảnh của người đó vào `collected_faces` (góc nghiêng, ánh sáng khác nhau) và chạy lại `process_collected_faces.py`.
