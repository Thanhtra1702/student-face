import os
import cv2
import shutil
import numpy as np
from deepface import DeepFace
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import sys
import datetime

sys.stdout.reconfigure(encoding='utf-8')

COLLECTED_DIR = "collected_faces"
PROCESSED_DIR = "collected_faces/processed"
DATABASE_DIR = "database"
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "student_faces"

def process_collected_images():
    if not os.path.exists(COLLECTED_DIR):
        print(f"Không tìm thấy thư mục {COLLECTED_DIR}")
        return

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # Lấy danh sách ảnh (bao gồm cả trong subfolders)
    image_files = []
    for root, dirs, files in os.walk(COLLECTED_DIR):
        # Bỏ qua thư mục 'processed'
        if 'processed' in root.replace('\\', '/').split('/'):
            continue
            
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                image_files.append(os.path.join(root, file))
    
    if not image_files:
        print("Không có ảnh nào cần xử lý.")
        return

    print(f"🔍 Tìm thấy {len(image_files)} ảnh cần xử lý...")
    
    # Init Qdrant Client một lần
    client = QdrantClient(path=DB_PATH)

    count_success = 0
    
    for file_path in image_files:
        filename = os.path.basename(file_path)
        
        # Parse MSSV từ tên file (Format: MSSV_Timestamp.jpg)
        try:
            mssv = filename.split('_')[0]
        except:
            print(f"⚠️ Tên file không đúng định dạng: {filename}")
            continue

        print(f"\n📸 Đang xử lý: {filename} (MSSV: {mssv})")
        
        # 1. Đọc và Crop ảnh
        img = cv2.imread(file_path)
        if img is None:
            print("❌ Lỗi đọc ảnh.")
            continue

        try:
            # ... (Phần Detection giữ nguyên) ...
            # Detect & Crop (Sử dụng logic thông minh: Diện tích + Vị trí trung tâm)
            try:
                results = DeepFace.represent(
                    img_path=img,
                    model_name="ArcFace",
                    enforce_detection=True,
                    detector_backend="mediapipe",
                    align=True
                )
            except:
                 results = DeepFace.represent(
                    img_path=img,
                    model_name="ArcFace",
                    enforce_detection=True,
                    detector_backend="opencv",
                    align=True
                )

            if not results:
                print("❌ Không phát hiện khuôn mặt.")
                continue

            # Logic chọn khuôn mặt tốt nhất
            img_height, img_width = img.shape[:2]
            img_center_x = img_width / 2
            img_center_y = img_height / 2
            
            best_face = None
            best_score = -1
            
            for face_data in results:
                fa = face_data['facial_area']
                area = fa['w'] * fa['h']
                face_center_x = fa['x'] + fa['w'] / 2
                face_center_y = fa['y'] + fa['h'] / 2
                distance = ((face_center_x - img_center_x)**2 + (face_center_y - img_center_y)**2)**0.5
                max_distance = (img_width**2 + img_height**2)**0.5
                distance_score = 1 - (distance / max_distance)
                max_area = img_width * img_height
                area_score = area / max_area
                total_score = 0.7 * area_score + 0.3 * distance_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_face = face_data
            
            # Crop & Save
            target_path = os.path.join(DATABASE_DIR, f"{mssv}.jpg")
            
            # Tính toán vùng crop có padding
            facial_area = best_face['facial_area']
            padding = 0.2
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            x_pad, y_pad = int(w * padding), int(h * padding)
            x1 = max(0, x - x_pad)
            y1 = max(0, y - y_pad)
            x2 = min(img_width, x + w + x_pad)
            y2 = min(img_height, y + h + y_pad)
            
            face_crop = img[y1:y2, x1:x2]
            cv2.imwrite(target_path, face_crop)
            print(f"✅ Đã crop và cập nhật database: {target_path}")

            # 2. Update Qdrant
            embedding = best_face['embedding']
            
            # Tìm ID hiện tại trong Qdrant
            try:
                scroll_result = client.scroll(
                    collection_name=COLLECTION_NAME,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="student_id",
                                match=models.MatchValue(value=mssv)
                            )
                        ]
                    ),
                    limit=1
                )[0]
                
                if scroll_result:
                    point_id = scroll_result[0].id
                    print(f"🔄 Đang cập nhật vector Qdrant (ID: {point_id})...")
                else:
                    import uuid
                    point_id = str(uuid.uuid4())
                    print(f"➕ Tạo vector mới Qdrant (ID: {point_id})...")
            except Exception as e:
                print(f"⚠️ Lỗi truy vấn Qdrant: {e}")
                # Fallback tạo mới nếu lỗi scroll
                import uuid
                point_id = str(uuid.uuid4())

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={"student_id": mssv}
                    )
                ]
            )
            print("✅ Đã cập nhật Qdrant.")

            # 3. Di chuyển ảnh gốc sang processed/MSSV/
            processed_student_dir = os.path.join(PROCESSED_DIR, mssv)
            if not os.path.exists(processed_student_dir):
                os.makedirs(processed_student_dir)
            
            shutil.move(file_path, os.path.join(processed_student_dir, filename))
            
            # Xóa thư mục rỗng trong collected_faces nếu cần (optional)
            parent_dir = os.path.dirname(file_path)
            if not os.listdir(parent_dir) and parent_dir != COLLECTED_DIR:
                os.rmdir(parent_dir)
                
            count_success += 1

        except Exception as e:
            print(f"❌ Lỗi xử lý {filename}: {e}")

    print("\n" + "="*50)
    print(f"🎉 Hoàn tất! Đã xử lý thành công {count_success}/{len(files)} ảnh.")

if __name__ == "__main__":
    process_collected_images()
