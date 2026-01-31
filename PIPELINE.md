# PIPELINE HOẠT ĐỘNG CỦA HỆ THỐNG FACE RECOGNITION

## 📋 CẤU TRÚC PROJECT

### ✅ FILES CORE (Bắt buộc)

```
main.py                 - Ứng dụng nhận diện khuôn mặt chính
init_qdrant.py          - Khởi tạo/Re-enroll vector database
student_info.db         - SQLite database thông tin sinh viên
requirements.txt        - Python dependencies
README.md               - Hướng dẫn sử dụng
Report.md               - Báo cáo dự án
.gitignore              - Git configuration
```

### ✅ FILES UTILITIES (Hỗ trợ)

```
crop_faces.py           - Crop khuôn mặt từ ảnh trong database/
sync_database_full.py   - Đồng bộ student_info.db với ảnh
test_accuracy.py        - Test độ chính xác hệ thống
setup_database.py       - Setup database lần đầu
crawl_fap.py            - [Reference] Script crawl ảnh từ FAP
```

### 📁 FOLDERS

```
database/               - Ảnh đã crop (1,447 ảnh) - SOURCE OF TRUTH
it_snotme/              - Backup ảnh gốc (1,585 ảnh)
flipped_faces/          - Ảnh test (16 ảnh)
qdrant_db/              - Vector database (embeddings)
venv/                   - Python virtual environment
```

---

## 🔄 PIPELINE HOẠT ĐỘNG

### 1️⃣ SETUP BAN ĐẦU (Chỉ chạy 1 lần)

```bash
# Bước 1: Cài đặt dependencies
pip install -r requirements.txt

# Bước 2: Chuẩn bị ảnh
# - Copy ảnh gốc vào it_snotme/
# - Copy ảnh test vào flipped_faces/

# Bước 3: Crop khuôn mặt từ ảnh gốc
python crop_faces.py
# Input: it_snotme/ + flipped_faces/
# Output: database_cropped/

# Bước 4: Copy ảnh đã crop vào database/
Copy-Item database_cropped/* database/

# Bước 5: Đồng bộ database
python sync_database_full.py
# Tạo student_info.db khớp với ảnh trong database/

# Bước 6: Khởi tạo Qdrant
python init_qdrant.py
# Trích xuất embeddings từ database/ → qdrant_db/
```

### 2️⃣ CHẠY ỨNG DỤNG

```bash
python main.py
```

**Pipeline nhận diện:**

```
Camera Frame
    ↓
[1] Phát hiện khuôn mặt (Mediapipe)
    ↓
[2] Kiểm tra khuôn mặt trong khung oval
    ↓
[3] Kiểm tra chất lượng (size, brightness, sharpness)
    ↓
[4] Crop khuôn mặt (loại bỏ background)
    ↓
[5] Trích xuất embedding (ArcFace) - Lặp 3 lần
    ↓
[6] Tính embedding trung bình (Multi-frame averaging)
    ↓
[7] Tìm kiếm trong Qdrant (Top 5)
    ↓
[8] Re-ranking thông minh:
    - Tier 1: Score > 0.60 → Accept
    - Tier 2: Score > 0.50 + (Ratio > 1.05 OR Gap > 0.04) → Accept
    - Tier 3: Score > 0.40 + (Ratio > 1.10 AND Gap > 0.06) → Accept
    - Tier 4: Score > 0.35 + (Ratio > 1.15 AND Gap > 0.10) → Accept
    - Else → Reject
    ↓
[9] Lấy thông tin từ student_info.db
    ↓
[10] Hiển thị kết quả
```

### 3️⃣ TEST ĐỘ CHÍNH XÁC

```bash
python test_accuracy.py
# Test trên 16 ảnh trong flipped_faces/
```

### 4️⃣ CẬP NHẬT DATABASE

**Khi thêm/xóa ảnh:**

```bash
# Bước 1: Cập nhật ảnh trong database/
# (Thêm/xóa file .jpg)

# Bước 2: Đồng bộ student_info.db
python sync_database_full.py

# Bước 3: Re-enroll Qdrant
python init_qdrant.py
```

**Khi cập nhật thông tin sinh viên:**

```bash
# Sửa trực tiếp trong student_info.db
# Hoặc dùng script SQL
```

---

## 🎯 CÔNG NGHỆ SỬ DỤNG

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Face Detection | Mediapipe | Phát hiện khuôn mặt + 6 landmarks |
| Face Alignment | DeepFace | Căn chỉnh khuôn mặt |
| Embedding Model | ArcFace | Trích xuất 512-dim vector |
| Vector Search | Qdrant | Tìm kiếm similarity |
| Database | SQLite | Lưu thông tin sinh viên |
| Preprocessing | CLAHE | Cải thiện contrast |
| UI | OpenCV + PIL | Hiển thị kết quả |

---

## 📊 THÔNG SỐ HỆ THỐNG

- **Số lượng sinh viên:** 1,447
- **Embedding dimension:** 512
- **Multi-frame:** 3 frames
- **Cooldown:** 2 giây
- **Resolution:** 640x480
- **FPS:** 30

---

## 🗑️ FILES CÓ THỂ XÓA

```bash
sync_db.py              # Thay bằng sync_database_full.py
check_multiple_faces.py # Script debug, đã dùng xong
test_results.txt        # Kết quả test cũ (68.75%)
database_cropped/       # Folder backup sau khi crop
```

**Chạy cleanup:**

```powershell
.\cleanup.ps1
```
