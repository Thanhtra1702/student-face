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

# --- AI ENHANCEMENT HELPERS (Đồng bộ với app.py và init_qdrant.py) ---
def preprocess_frame(frame):
    """Cân bằng sáng và khử nhiễu để AI dễ đọc hơn"""
    try:
        denoised = cv2.GaussianBlur(frame, (3, 3), 0)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
    except:
        return frame

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))
# -----------------------------------------------------------------------

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
        parent_dir_name = os.path.basename(os.path.dirname(file_path))
        
        # LOGIC CHUẨN: Chỉ nhận ảnh trong Folder con (collected_faces/MSSV/...)
        if parent_dir_name == "collected_faces" or parent_dir_name == "processed":
            print(f"⚠️ Bỏ qua ảnh không nằm trong thư mục MSSV: {filename}")
            continue
            
        mssv = parent_dir_name
        
        print(f"\n📸 Đang xử lý: {filename} (MSSV: {mssv})")
        
        # 1. Đọc và Tiền xử lý (Đồng bộ với app.py)
        img_raw = cv2.imread(file_path)
        if img_raw is None:
            print("❌ Lỗi đọc ảnh.")
            continue
        
        img = preprocess_frame(img_raw)

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

            # 2. Update Qdrant (Cơ chế Multi-Vector + Augmentation: Tạo x8 variants)
            # Tạo các biến thể (Augmentation - Buff mạnh để tăng độ chính xác)
            # Dùng cv2.convertScaleAbs cho Brightness/Contrast
            variants = [
                ("orig", face_crop),
                ("flip", cv2.flip(face_crop, 1)),
                ("rot_p5", rotate_image(face_crop, 5)),
                ("rot_m5", rotate_image(face_crop, -5)),
                ("bright", cv2.convertScaleAbs(face_crop, alpha=1.2, beta=30)), # Sáng hơn
                ("dark", cv2.convertScaleAbs(face_crop, alpha=0.8, beta=-20)),   # Tối hơn
                ("contrast", cv2.convertScaleAbs(face_crop, alpha=1.5, beta=0)), # Tương phản cao
                ("blur", cv2.GaussianBlur(face_crop, (3, 3), 0))                # Nhòe nhẹ
            ]
            
            import uuid
            for var_name, var_img in variants:
                try:
                    # Chuyển sang RGB trước khi xử lý
                    rgb_var = cv2.cvtColor(var_img, cv2.COLOR_BGR2RGB)
                    
                    results_var = DeepFace.represent(
                        img_path=rgb_var,
                        model_name="ArcFace",
                        enforce_detection=False,
                        detector_backend="mediapipe",
                        align=True
                    )
                    if results_var:
                        embedding = results_var[0]['embedding']
                        point_id = str(uuid.uuid4())
                        client.upsert(
                            collection_name=COLLECTION_NAME,
                            points=[
                                PointStruct(
                                    id=point_id,
                                    vector=embedding,
                                    payload={"student_id": mssv, "variant": var_name}
                                )
                            ]
                        )
                except:
                    pass
            
            print(f"✅ Đã thêm 4 variants vào Qdrant cho {mssv}.")

            # 3. Lưu trữ: Di chuyển ảnh vào thư mục processed thay vì xóa (để đối soát)
            try:
                # Tạo cấu trúc thư mục MSSV bên trong processed
                dest_dir = os.path.join(PROCESSED_DIR, mssv)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                
                # Di chuyển file (thêm timestamp để tránh trùng tên nếu 1 MSSV có nhiều ảnh)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                dest_path = os.path.join(dest_dir, f"{timestamp}_{filename}")
                
                shutil.move(file_path, dest_path)
                print(f"📦 Đã lưu trữ ảnh gốc vào: {dest_path}")
            except Exception as e:
                print(f"⚠️ Không thể lưu trữ file {filename}: {e}")
            
            # Xóa thư mục rỗng trong collected_faces nếu cần
            parent_dir = os.path.dirname(file_path)
            if not os.listdir(parent_dir) and parent_dir != COLLECTED_DIR:
                try:
                    os.rmdir(parent_dir)
                except: pass
                
            count_success += 1

        except Exception as e:
            print(f"❌ Lỗi xử lý {filename}: {e}")

    print("\n" + "="*50)
    print(f"🎉 Hoàn tất! Đã xử lý thành công {count_success}/{len(image_files)} ảnh.")

if __name__ == "__main__":
    process_collected_images()
