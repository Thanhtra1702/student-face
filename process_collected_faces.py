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
            
            # --- LOGIC CHỌN ẢNH TỐT NHẤT (SMART AVATAR SELECTION) ---
            should_save_image = True
            if os.path.exists(target_path):
                # Nếu ảnh đã tồn tại, so sánh chất lượng (Dựa trên độ phân giải)
                try:
                    old_img = cv2.imread(target_path)
                    if old_img is not None:
                        old_h, old_w = old_img.shape[:2]
                        new_h, new_w = face_crop.shape[:2]
                        
                        old_area = old_w * old_h
                        new_area = new_w * new_h
                        
                        # Chỉ thay thế nếu ảnh mới LỚN HƠN ảnh cũ (Rõ nét hơn)
                        if new_area <= old_area:
                            should_save_image = False
                            print(f"ℹ️ Giữ nguyên Avatar cũ (Mới: {new_w}x{new_h} <= Cũ: {old_w}x{old_h})")
                        else:
                            print(f"🆙 Cập nhật Avatar chất lượng cao hơn ({old_w}x{old_h} -> {new_w}x{new_h})")
                except:
                    pass # Lỗi đọc ảnh cũ -> Cứ ghi đè cho chắc

            if should_save_image:
                cv2.imwrite(target_path, face_crop)
                print(f"✅ Đã lưu Avatar mới: {target_path}")
            # --------------------------------------------------------

            # 2. Update Qdrant (Cơ chế Multi-Vector: Luôn tạo điểm mới)
            embedding = best_face['embedding']
            
            # Tạo ID ngẫu nhiên cho Vector mới (Không ghi đè Vector cũ)
            import uuid
            point_id = str(uuid.uuid4())
            print(f"➕ Thêm dữ liệu học mới cho {mssv} (Point ID: {point_id})...")

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
            print("✅ Đã nạp thêm vào Qdrant.")

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
