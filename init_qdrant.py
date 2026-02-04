import os
import sys
import cv2  # Added
import numpy as np # Added
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from deepface import DeepFace
from tqdm import tqdm

# Cấu hình
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "student_faces"
IMAGE_DIR = "./database"
MODEL_NAME = "ArcFace"

# --- AI ENHANCEMENT HELPERS (Copy từ app.py để đồng bộ) ---
def preprocess_frame(frame):
    """Cân bằng sáng và khử nhiễu để AI dễ đọc hơn"""
    try:
        # 1. Khử nhiễu nhẹ
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)
        
        # 2. Chuyển sang LAB để cân bằng sáng (CLAHE)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
    except:
        return frame
# Helper xoay ảnh
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

def init_qdrant():
    # Khởi tạo client Qdrant lưu trữ local
    client = QdrantClient(path=DB_PATH)

    # Tạo lại collection
    print(f"Đang tạo lại collection '{COLLECTION_NAME}' với model {MODEL_NAME}...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("Không tìm thấy ảnh nào trong thư mục database!")
        return

    print(f"Bắt đầu xử lý {len(image_files)} ảnh (Augmentation x8 variants)...")
    
    points = []
    import uuid

    for filename in tqdm(image_files):
        student_id = os.path.splitext(filename)[0]
        img_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            # 1. Đọc ảnh gốc
            img = cv2.imread(img_path)
            if img is None: continue
            
            # 2. Tạo các biến thể (Augmentation - Buff x8 để tăng độ chính xác)
            # Dùng cv2.convertScaleAbs cho Brightness/Contrast
            variants = [
                ("orig", img),
                ("flip", cv2.flip(img, 1)),
                ("rot_p5", rotate_image(img, 5)),
                ("rot_m5", rotate_image(img, -5)),
                ("bright", cv2.convertScaleAbs(img, alpha=1.2, beta=30)), # Sáng hơn
                ("dark", cv2.convertScaleAbs(img, alpha=0.8, beta=-20)),   # Tối hơn
                ("contrast", cv2.convertScaleAbs(img, alpha=1.5, beta=0)), # Tương phản cao
                ("blur", cv2.GaussianBlur(img, (3, 3), 0))                # Nhòe nhẹ (sensor noise)
            ]
            
            # 3. Tạo vector cho từng biến thể
            for var_name, var_img in variants:
                try:
                    # Chuyển sang RGB trước khi đưa vào DeepFace
                    rgb_var = cv2.cvtColor(var_img, cv2.COLOR_BGR2RGB)
                    
                    results = DeepFace.represent(
                        img_path=rgb_var, 
                        model_name=MODEL_NAME, 
                        detector_backend="mediapipe",
                        align=True,
                        enforce_detection=False
                    )
                    
                    if results:
                        embedding = results[0]["embedding"]
                        points.append(PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                "student_id": student_id, 
                                "filename": filename,
                                "variant": var_name
                            }
                        ))
                except:
                    pass
                    
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {e}")

    # Push dữ liệu lên Qdrant
    if points:
        # Chia nhỏ batches để tránh lỗi memory nếu quá nhiều
        batch_size = 100
        print(f"Đang lưu {len(points)} vector vào DB...")
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
        print(f"✅ Đã lưu {len(points)} vector vào Qdrant thành công!")
    else:
        print("Không có vector nào được tạo.")

if __name__ == "__main__":
    init_qdrant()
