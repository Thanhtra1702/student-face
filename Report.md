# 📊 BÁO CÁO DỰ ÁN: HỆ THỐNG KIOSK NHẬN DIỆN KHUÔN MẶT

**Tên dự án:** Face Recognition Kiosk System  
**Ngày hoàn thành:** 31/01/2026  
**Công nghệ chính:** ArcFace, Mediapipe, Qdrant Vector Database  
**Repository:** <https://github.com/Thanhtra1702/Face-Recognition.git>

---

## 📑 MỤC LỤC

1. [Tổng Quan](#1-tổng-quan)
2. [Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
3. [Công Nghệ Sử Dụng](#3-công-nghệ-sử-dụng)
4. [Quy Trình Nhận Diện](#4-quy-trình-nhận-diện)
5. [Tối Ưu Hóa](#5-tối-ưu-hóa)
6. [Hướng Dẫn Sử Dụng](#6-hướng-dẫn-sử-dụng)
7. [Hạn Chế và Phát Triển](#7-hạn-chế-và-phát-triển)

---

## 1. TỔNG QUAN

### 1.1. Mục Tiêu

Xây dựng hệ thống Kiosk tự động nhận diện khuôn mặt sinh viên để tra cứu lịch học, giúp:

- Tăng tốc độ tra cứu (không cần nhập MSSV)
- Nâng cao trải nghiệm người dùng
- Tự động hóa quy trình điểm danh

### 1.2. Thông Số Hệ Thống

- **Quy mô:** 1,604 sinh viên
- **Thời gian nhận diện:** ~3 giây/người
- **Môi trường:** Kiosk tại các điểm công cộng

### 1.3. Đặc Điểm Nổi Bật

✅ Model ArcFace state-of-the-art (99.83% accuracy trên LFW)  
✅ Multi-frame averaging giảm nhiễu  
✅ Re-ranking thông minh với Confidence Ratio  
✅ CLAHE preprocessing cho điều kiện ánh sáng kém  
✅ Qdrant vector database tìm kiếm nhanh O(log n)  

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1. Sơ Đồ Tổng Quan

```
Camera (640x480)
    ↓
Face Detection (Mediapipe)
    ↓
Preprocessing (CLAHE)
    ↓
Multi-frame Averaging (3 embeddings)
    ↓
Embedding Extraction (ArcFace)
    ↓
Vector Search (Qdrant)
    ↓
Re-ranking (Confidence Ratio)
    ↓
Info Retrieval (SQLite)
    ↓
Display Results
```

### 2.2. Các Thành Phần Chính

| Thành phần | Công nghệ | Chức năng |
|------------|-----------|-----------|
| **Face Detection** | Mediapipe | Phát hiện 6 landmarks, căn chỉnh khuôn mặt |
| **Preprocessing** | CLAHE | Cải thiện độ tương phản |
| **Embedding** | ArcFace | Trích xuất vector 512 chiều |
| **Vector DB** | Qdrant | Lưu trữ và tìm kiếm embeddings |
| **Info DB** | SQLite | Lưu trữ thông tin sinh viên |

---

## 3. CÔNG NGHỆ SỬ DỤNG

### 3.1. Deep Learning Models

#### **ArcFace**

- **Độ chính xác:** 99.83% trên LFW dataset
- **Output:** Vector 512 chiều
- **Ưu điểm:** Vượt trội FaceNet, VGG-Face trong môi trường đông người

#### **Mediapipe**

- **Landmarks:** 6 điểm (2 mắt, mũi, 2 góc miệng, trán)
- **Tốc độ:** Real-time (>30 FPS)
- **Ưu điểm:** Căn chỉnh khuôn mặt chính xác hơn OpenCV

### 3.2. Image Processing

#### **CLAHE (Contrast Limited Adaptive Histogram Equalization)**

- **Tham số:** Clip Limit 2.0, Tile Grid 8×8
- **Mục đích:** Cải thiện ảnh trong điều kiện ánh sáng kém
- **Hiệu quả:** Tăng độ rõ nét các chi tiết khuôn mặt

### 3.3. Database

#### **Qdrant Vector Database**

- **Thuật toán:** HNSW (Hierarchical Navigable Small World)
- **Metric:** Cosine Similarity
- **Độ phức tạp:** O(log n)
- **Quy mô:** 1,604 vectors × 512 dimensions

#### **SQLite**

- **Bảng:** students (id, name, schedule, room)
- **Records:** 1,604 sinh viên

### 3.4. Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| Deep Learning | DeepFace | ≥0.0.79 |
| Computer Vision | OpenCV | ≥4.5.0 |
| Vector DB | Qdrant | Latest |
| Framework | TensorFlow | ≥2.15.0 |

---

## 4. QUY TRÌNH NHẬN DIỆN

### 4.1. Pipeline (6 Giai Đoạn)

**Giai đoạn 1: Face Detection**

- Mediapipe phát hiện khuôn mặt và 6 landmarks
- Căn chỉnh khuôn mặt về tư thế chuẩn

**Giai đoạn 2: Quality Check**

- Kích thước: ≥ 80×80 pixels
- Độ sáng: [40, 220]
- Độ nét: Laplacian variance ≥ 100

**Giai đoạn 3: Preprocessing**

- CLAHE enhancement trên L channel (LAB color space)
- Cải thiện độ tương phản cục bộ

**Giai đoạn 4: Multi-frame Averaging**

- Lấy 3 embeddings từ cùng 1 khuôn mặt
- Thêm nhiễu ngẫu nhiên nhẹ để tạo đa dạng
- Tính embedding trung bình

**Giai đoạn 5: Vector Search**

- Tìm kiếm top-5 trong Qdrant
- Sử dụng cosine similarity

**Giai đoạn 6: Re-ranking**

- Tính Confidence Gap và Ratio
- Áp dụng 4-tier decision logic

### 4.2. Decision Logic (4 Tiers)

| Tier | Score Range | Điều kiện | Accuracy |
|------|-------------|-----------|----------|
| **1** | > 0.60 | Chấp nhận ngay | 99%+ |
| **2** | 0.50-0.60 | Ratio > 1.05 HOẶC Gap > 0.04 | 95%+ |
| **3** | 0.40-0.50 | Ratio > 1.10 VÀ Gap > 0.06 | 85%+ |
| **4** | 0.35-0.40 | Ratio > 1.15 VÀ Gap > 0.10 | 75%+ |
| **Reject** | < 0.35 | Từ chối | - |

**Confidence Ratio** = Score_1 / Score_2  
**Confidence Gap** = Score_1 - Score_2

---

## 5. TỐI ƯU HÓA

### 5.1. Các Kỹ Thuật Áp Dụng

| Kỹ thuật | Cải thiện | Mô tả |
|----------|-----------|-------|
| **Mediapipe Detector** | +10-15% | Căn chỉnh chính xác với 6 landmarks |
| **CLAHE Preprocessing** | +5-8% | Cải thiện trong điều kiện ánh sáng kém |
| **Multi-frame Averaging** | +5-10% | Giảm nhiễu ngẫu nhiên |
| **Confidence Ratio** | +8-12% | Phân biệt tốt hơn so với chỉ dùng Gap |
| **Quality Check** | +3-5% | Loại bỏ ảnh kém chất lượng |

---

## 6. HƯỚNG DẪN SỬ DỤNG

### 6.1. Yêu Cầu Hệ Thống

- **OS:** Windows 10/11, macOS, Linux
- **Python:** 3.10+
- **RAM:** 8GB (khuyến nghị 16GB)
- **Storage:** ~5GB
- **Camera:** Webcam/USB camera

### 7.2. Cài Đặt

**Bước 1: Clone Repository**

```bash
git clone https://github.com/Thanhtra1702/Face-Recognition.git
cd Face-Recognition
```

**Bước 2: Tạo Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

**Bước 3: Cài Dependencies**

```bash
pip install -r requirements.txt
```

**Bước 4: Khởi tạo Database**

```bash
# Thêm ảnh vào database/ (tên file: MSSV.jpg)
python setup_database.py
python init_qdrant.py
```

**Bước 5: Chạy**

```bash
python main.py
```

### 7.3. Cấu Trúc Dự Án

```
student_face/
├── main.py                # Ứng dụng Kiosk chính
├── init_qdrant.py        # Khởi tạo Qdrant DB
├── setup_database.py     # Tạo SQLite DB
├── sync_db.py            # Đồng bộ database
├── requirements.txt      # Dependencies
├── README.md             # Hướng dẫn cơ bản
├── Report.md             # Báo cáo chi tiết
├── database/             # Ảnh sinh viên (1604 ảnh)
├── qdrant_db/            # Vector database
└── student_info.db       # SQLite database
```

---

## 7. HẠN CHẾ VÀ PHÁT TRIỂN

### 7.1. Hạn Chế Hiện Tại

**1. Nhiễu Database**

- Còn ảnh với nhãn `0 (x)` gây False Positive
- **Giải pháp:** Làm sạch database hoặc logic ưu tiên MSSV thật

**2. Điều Kiện Ánh Sáng**

- Giảm accuracy trong ánh sáng cực kém/gắt
- **Giải pháp:** Đèn LED ổn định tại Kiosk

**3. Góc Nhìn**

- Kém khi nghiêng mặt >30°
- **Giải pháp:** Hướng dẫn trực quan

**4. Tốc Độ**

- 3s/người có thể chậm giờ cao điểm
- **Giải pháp:** Giảm frame averaging 3→2

### 7.3. Tính Năng Tương Lai

✨ **Thống kê** - Dashboard số lượt tra cứu  
✨ **Multi-language** - Hỗ trợ tiếng Anh  
✨ **Voice feedback** - Đọc tên bằng giọng nói  

---

## 📚 TÀI LIỆU THAM KHẢO

1. **ArcFace:** Deng, J., et al. (2019). ArcFace: Additive Angular Margin Loss for Deep Face Recognition. CVPR 2019.

2. **Mediapipe:** Google Research. MediaPipe Face Detection. <https://google.github.io/mediapipe/>

3. **Qdrant:** Qdrant Vector Database Documentation. <https://qdrant.tech/>

4. **CLAHE:** Pizer, S. M., et al. (1987). Adaptive Histogram Equalization.

---

**Ngày cập nhật:** 31/01/2026  
