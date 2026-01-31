import os
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from deepface import DeepFace
from tqdm import tqdm

# Cấu hình
DB_PATH = "./qdrant_db"#
# DB_URL = "http://localhost:6333"
COLLECTION_NAME = "student_faces"
IMAGE_DIR = "./database"
MODEL_NAME = "ArcFace"  # Model dimension = 512, độ chính xác cao nhất

def init_qdrant():
    # Khởi tạo client Qdrant lưu trữ local
    client = QdrantClient(path=DB_PATH)

    # Tạo lại collection (Xóa cũ tạo mới để cập nhật Model mới)
    print(f"Đang tạo lại collection '{COLLECTION_NAME}' với model {MODEL_NAME}...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    # Lấy danh sách ảnh trong thư mục database
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("Không tìm thấy ảnh nào trong thư mục database!")
        return

    print(f"Bắt đầu xử lý {len(image_files)} ảnh...")
    
    points = []
    for i, filename in enumerate(tqdm(image_files)):
        student_id = os.path.splitext(filename)[0]
        img_path = os.path.join(IMAGE_DIR, filename)
        
        try:
            # Tạo embedding cho khuôn mặt
            # enforce_detection=False để tránh lỗi nếu ảnh không rõ mặt
            results = DeepFace.represent(img_path=img_path, model_name=MODEL_NAME, enforce_detection=False)
            
            if results:
                embedding = results[0]["embedding"]
                
                # Tạo point để push lên Qdrant
                points.append(PointStruct(
                    id=i,
                    vector=embedding,
                    payload={"student_id": student_id, "filename": filename}
                ))
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {e}")

    # Push dữ liệu lên Qdrant
    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        print(f"✅ Đã lưu {len(points)} vector vào Qdrant!")
    else:
        print("Không có vector nào được tạo.")

if __name__ == "__main__":
    init_qdrant()
