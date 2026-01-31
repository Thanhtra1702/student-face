# 🎓 Hệ Thống Tra Cứu Lịch Học Sinh Viên Bằng Nhận Diện Khuôn Mặt

## 📁 Cấu Trúc Thư Mục

```
student_face/
├── main.py              # File chạy chính
├── setup_database.py    # Script tạo database mẫu
├── requirements.txt     # Các thư viện cần thiết
├── student_info.db      # SQLite database (tạo sau khi chạy setup)
├── database/            # Thư mục chứa ảnh thẻ sinh viên
│   ├── 21120001.jpg     # Ảnh đặt tên theo MSSV
│   ├── 21120002.jpg
│   └── ...
└── README.md
```

## 🚀 Hướng Dẫn Cài Đặt

### 1. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 2. Khởi tạo database
```bash
python setup_database.py
```

### 3. Thêm ảnh sinh viên
- Đặt ảnh thẻ sinh viên vào thư mục `./database/`
- **Quan trọng**: Đặt tên file theo MSSV (ví dụ: `21120001.jpg`)

### 4. Chạy chương trình
```bash
python main.py
```

## 🎮 Hướng Dẫn Sử Dụng

- **Nhấn 's'**: Scan khuôn mặt để tra cứu lịch học
- **Nhấn 'q'**: Thoát chương trình

## ⚙️ Công Nghệ Sử Dụng

- **DeepFace**: Thư viện nhận diện khuôn mặt với model ArcFace
- **OpenCV**: Xử lý ảnh và camera
- **SQLite**: Database lưu trữ thông tin sinh viên
- **Pandas**: Xử lý dữ liệu

## 📝 Ghi Chú

- Đảm bảo camera hoạt động trước khi chạy
- Ảnh trong `database/` nên là ảnh chính diện, rõ mặt
- Lần đầu chạy, DeepFace sẽ tải model ArcFace (~500MB)
