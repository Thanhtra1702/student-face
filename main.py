import cv2
import os
from deepface import DeepFace
import sqlite3
import pandas as pd
from qdrant_client import QdrantClient
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 1. Khởi tạo Qdrant Client và Database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "student_faces"

# Khởi tạo client
client = QdrantClient(path=QDRANT_PATH)

def get_student_info(student_id):
    conn = sqlite3.connect(os.path.join(BASE_DIR, 'student_info.db'))
    query = f"SELECT name, schedule, room FROM students WHERE id = '{student_id}'"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def put_vietnamese_text(img, text, position, font_size=32, color=(0, 255, 0)):
    """Vẽ text tiếng Việt lên frame sử dụng PIL"""
    # Chuyển từ BGR (OpenCV) sang RGB (PIL)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Sử dụng font mặc định của Windows hỗ trợ tiếng Việt
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Vẽ text
    draw.text(position, text, font=font, fill=color)
    
    # Chuyển lại sang BGR (OpenCV)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def get_largest_face(faces_data):
    """Chọn khuôn mặt lớn nhất (gần camera nhất) từ danh sách khuôn mặt"""
    if not faces_data:
        return None
    
    largest_face = None
    max_area = 0
    
    for face in faces_data:
        if 'facial_area' in face:
            area = face['facial_area']
            face_area = area['w'] * area['h']
            if face_area > max_area:
                max_area = face_area
                largest_face = face
    
    return largest_face if largest_face else faces_data[0]

def is_face_in_frame(face_area, frame_center, frame_size, scale=2):
    """Kiểm tra xem khuôn mặt có nằm trong khung oval không"""
    # Scale lại tọa độ vì đã resize
    face_x = face_area['x'] * scale
    face_y = face_area['y'] * scale
    face_w = face_area['w'] * scale
    face_h = face_area['h'] * scale
    
    # Tính tâm khuôn mặt
    face_center_x = face_x + face_w // 2
    face_center_y = face_y + face_h // 2
    
    # Tính khoảng cách từ tâm khuôn mặt đến tâm khung
    dx = abs(face_center_x - frame_center[0])
    dy = abs(face_center_y - frame_center[1])
    
    # Kiểm tra xem tâm khuôn mặt có nằm trong ellipse không
    # Công thức ellipse: (x/a)^2 + (y/b)^2 <= 1
    a = frame_size[0] // 2  # Bán trục ngang
    b = frame_size[1] // 2  # Bán trục dọc
    
    return (dx / a) ** 2 + (dy / b) ** 2 <= 1

def draw_face_guide_frame(frame, progress=0):
    """Vẽ khung hướng dẫn hình oval ở giữa màn hình với progress bar"""
    h, w = frame.shape[:2]
    
    # Kích thước khung oval (tăng lên 50% chiều rộng và 70% chiều cao)
    oval_w = int(w * 0.5)
    oval_h = int(h * 0.7)
    
    # Tâm khung
    center_x = w // 2
    center_y = h // 2
    
    if progress > 0:
        # Vẽ progress bar (arc màu xanh chạy quanh oval)
        # progress từ 0 đến 100
        angle = int(360 * progress / 100)
        cv2.ellipse(frame, (center_x, center_y), (oval_w // 2, oval_h // 2), 
                    0, -90, -90 + angle, (0, 255, 0), 8)  # Màu xanh, độ dày 8
        
        # Vẽ phần còn lại màu trắng mờ
        if angle < 360:
            cv2.ellipse(frame, (center_x, center_y), (oval_w // 2, oval_h // 2), 
                        0, -90 + angle, 270, (255, 255, 255), 3)
    else:
        # Vẽ ellipse trắng bình thường khi chưa có progress
        cv2.ellipse(frame, (center_x, center_y), (oval_w // 2, oval_h // 2), 
                    0, 0, 360, (255, 255, 255), 3)
    
    # Vẽ text hướng dẫn
    if progress == 0:
        frame = put_vietnamese_text(frame, "Đặt khuôn mặt vào khung", 
                                    (center_x - 150, center_y - oval_h // 2 - 50), 
                                    font_size=24, color=(255, 255, 255))
    elif progress < 100:
        frame = put_vietnamese_text(frame, "Đang quét...", 
                                    (center_x - 80, center_y - oval_h // 2 - 50), 
                                    font_size=28, color=(0, 255, 0))
    
    return frame, (center_x, center_y), (oval_w, oval_h)

def draw_student_info(frame, name, student_id):
    """Vẽ thông tin sinh viên ở góc trên bên trái (nhỏ gọn)"""
    # Vẽ background semi-transparent cho text (nhỏ hơn)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (320, 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Vẽ thông tin với font nhỏ hơn
    frame = put_vietnamese_text(frame, f"MSSV: {student_id}", 
                                (20, 25), font_size=18, color=(0, 255, 0))
    frame = put_vietnamese_text(frame, f"{name}", 
                                (20, 55), font_size=22, color=(255, 255, 255))
    
    return frame

# 2. Khởi tạo Camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Giảm resolution để tăng tốc
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Hệ thống Kiosk FPT đang sẵn sàng. Đang quét khuôn mặt...")
print("Đặt khuôn mặt vào khung oval để tự động nhận diện | Nhấn 'q' để thoát")

frame_count = 0
display_frame = None
last_scan_time = 0
scan_cooldown = 2  # Thời gian chờ giữa các lần scan (giây) - giảm xuống 2s
face_detected_time = None  # Thời điểm phát hiện khuôn mặt
countdown_duration = 3  # 3 giây countdown
is_recognizing = False  # Đang trong quá trình nhận diện
recognition_result = None  # Lưu kết quả nhận diện
recognition_started = False  # Đã bắt đầu nhận diện chưa

import time
import threading

def check_face_quality(frame, face_area):
    """Kiểm tra chất lượng khuôn mặt"""
    try:
        x, y, w, h = face_area['x'], face_area['y'], face_area['w'], face_area['h']
        
        # Crop khuôn mặt
        face_crop = frame[y:y+h, x:x+w]
        
        if face_crop.size == 0:
            return False, "Khuôn mặt quá nhỏ"
        
        # 1. Kiểm tra kích thước (phải đủ lớn)
        if w < 80 or h < 80:
            return False, "Khuôn mặt quá nhỏ"
        
        # 2. Kiểm tra độ sáng
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if brightness < 40 or brightness > 220:
            return False, "Ánh sáng không phù hợp"
        
        # 3. Kiểm tra độ nét (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return False, "Ảnh bị mờ"
        
        return True, "OK"
    except:
        return False, "Lỗi kiểm tra chất lượng"

def recognize_face_async(small_frame, frame_center, frame_size):
    """Nhận diện khuôn mặt với multi-frame averaging và re-ranking thông minh"""
    global recognition_result, is_recognizing
    
    try:
        # Multi-frame averaging: Lấy 3 embeddings từ các frame khác nhau
        embeddings_list = []
        
        for attempt in range(3):
            # Preprocessing CLAHE
            lab = cv2.cvtColor(small_frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            frame_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Thêm một chút nhiễu ngẫu nhiên để tạo sự đa dạng giữa các frame
            if attempt > 0:
                noise = np.random.normal(0, 2, frame_enhanced.shape).astype(np.uint8)
                frame_enhanced = cv2.add(frame_enhanced, noise)
            
            # Detector Mediapipe
            try:
                results = DeepFace.represent(
                    img_path=frame_enhanced, 
                    model_name="ArcFace",
                    enforce_detection=True,
                    detector_backend="mediapipe",
                    align=True
                )
            except:
                try:
                    results = DeepFace.represent(
                        img_path=small_frame, 
                        model_name="ArcFace",
                        enforce_detection=True,
                        detector_backend="opencv",
                        align=True
                    )
                except:
                    results = None
            
            if results:
                faces_in_frame = []
                for face in results:
                    if 'facial_area' in face:
                        if is_face_in_frame(face['facial_area'], frame_center, frame_size, scale=1.0):
                            quality_ok, quality_msg = check_face_quality(small_frame, face['facial_area'])
                            if quality_ok:
                                faces_in_frame.append(face)
                
                if faces_in_frame:
                    selected_face = get_largest_face(faces_in_frame) if len(faces_in_frame) > 1 else faces_in_frame[0]
                    embeddings_list.append(selected_face["embedding"])
        
        # Nếu không lấy được embedding nào
        if len(embeddings_list) == 0:
            recognition_result = {
                'success': False,
                'message': "Không phát hiện được khuôn mặt hoặc chất lượng không đủ."
            }
            is_recognizing = False
            return
        
        # Tính embedding trung bình
        avg_embedding = np.mean(embeddings_list, axis=0).tolist()
        
        # Tìm kiếm với embedding trung bình
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=avg_embedding,
            limit=5  # Lấy top 5 để re-ranking
        ).points
        
        if search_result and len(search_result) > 0:
            best_match = search_result[0]
            
            # Re-ranking với Confidence Ratio
            confidence_gap = 0
            confidence_ratio = 1.0
            
            if len(search_result) > 1:
                confidence_gap = best_match.score - search_result[1].score
                # Tránh chia cho 0
                if search_result[1].score > 0:
                    confidence_ratio = best_match.score / search_result[1].score
            
            # Logic nhận diện nâng cao với Confidence Ratio
            accept = False
            
            # Tier 1: Score rất cao (>0.60) - Chấp nhận ngay
            if best_match.score > 0.60:
                accept = True
            # Tier 2: Score cao (0.50-0.60) - Cần ratio > 1.05 HOẶC gap > 0.04
            elif best_match.score > 0.50:
                if confidence_ratio > 1.05 or confidence_gap > 0.04:
                    accept = True
            # Tier 3: Score trung bình (0.40-0.50) - Cần ratio > 1.10 VÀ gap > 0.06
            elif best_match.score > 0.40:
                if confidence_ratio > 1.10 and confidence_gap > 0.06:
                    accept = True
            # Tier 4: Score thấp (0.35-0.40) - Cần ratio > 1.15 VÀ gap > 0.10
            elif best_match.score > 0.35:
                if confidence_ratio > 1.15 and confidence_gap > 0.10:
                    accept = True
            
            if accept:
                student_id = best_match.payload["student_id"]
                score = best_match.score
                info = get_student_info(student_id)
                
                if not info.empty:
                    recognition_result = {
                        'success': True,
                        'student_id': student_id,
                        'score': score,
                        'name': info.iloc[0]['name'],
                        'schedule': info.iloc[0]['schedule'],
                        'room': info.iloc[0]['room']
                    }
                else:
                    recognition_result = {
                        'success': False,
                        'message': f"Nhận diện được {student_id} nhưng chưa có dữ liệu lý lịch."
                    }
            else:
                if best_match.score < 0.35:
                    recognition_result = {
                        'success': False,
                        'message': f"Độ tin cậy quá thấp ({best_match.score:.2f})."
                    }
                else:
                    recognition_result = {
                        'success': False,
                        'message': f"Không thể xác định (Score: {best_match.score:.2f}, Gap: {confidence_gap:.2f}, Ratio: {confidence_ratio:.2f})."
                    }
        else:
            recognition_result = {
                'success': False,
                'message': "Không tìm thấy thông tin sinh viên phù hợp."
            }
    
    except Exception as e:
        recognition_result = {'success': False, 'message': f"Lỗi: {str(e)}"}
    
    is_recognizing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    current_time = time.time()
    
    frame = cv2.flip(frame, 1)
    # Tạo bản sao để vẽ
    display_frame = frame.copy()
    
    # Kiểm tra cooldown
    if current_time - last_scan_time < scan_cooldown:
        # Đang trong thời gian chờ, chỉ hiển thị khung
        display_frame, frame_center, frame_size = draw_face_guide_frame(display_frame)
        cv2.imshow("Kiosk Tra Cuu Lich Hoc FPT", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue
    
    # Nếu đang trong quá trình countdown
    if face_detected_time is not None:
        elapsed = current_time - face_detected_time
        remaining = countdown_duration - elapsed
        
        # Bắt đầu nhận diện ngay khi countdown bắt đầu (chỉ 1 lần)
        if not recognition_started and not is_recognizing:
            recognition_started = True
            is_recognizing = True
            recognition_result = None
            
            # Nhận diện với ảnh full resolution để tăng độ chính xác
            # Không resize, dùng ảnh gốc 640x480
            display_frame, frame_center, frame_size = draw_face_guide_frame(display_frame)
            
            thread = threading.Thread(target=recognize_face_async, 
                                     args=(frame, frame_center, frame_size))
            thread.daemon = True
            thread.start()
        
        if remaining > 0:
            # Vẽ countdown và progress bar mượt mà
            progress = int((elapsed / countdown_duration) * 100)
            display_frame, frame_center, frame_size = draw_face_guide_frame(display_frame, progress)
            
            # Hiển thị số đếm ngược
            countdown_text = f"{int(remaining) + 1}"
            h, w = display_frame.shape[:2]
            display_frame = put_vietnamese_text(display_frame, countdown_text,
                                                (w//2 - 20, h//2), 
                                                font_size=60, color=(0, 255, 0))
            cv2.imshow("Kiosk Tra Cuu Lich Hoc FPT", display_frame)
        else:
            # Hết countdown - Hiển thị kết quả
            if recognition_result is not None:
                if recognition_result['success']:
                    print(f"\n✅ Nhận diện được MSSV: {recognition_result['student_id']} (Độ tin cậy: {recognition_result['score']:.2f})")
                    print(f"--- CHÀO MỪNG SINH VIÊN ---")
                    print(f"Họ tên: {recognition_result['name']}")
                    print(f"Lịch học hôm nay:")
                    print(f"📌 Môn: {recognition_result['schedule']}, Phòng: {recognition_result['room']}")
                    print("----------------------------")
                    
                    # Hiển thị thông tin
                    result_frame = frame.copy()
                    result_frame, _, _ = draw_face_guide_frame(result_frame, 100)
                    result_frame = draw_student_info(result_frame, 
                                                    recognition_result['name'], 
                                                    recognition_result['student_id'])
                    cv2.imshow("Kiosk Tra Cuu Lich Hoc FPT", result_frame)
                    cv2.waitKey(3000)  # Hiển thị 3 giây
                    
                    last_scan_time = time.time()
                else:
                    print(f"⚠️ {recognition_result['message']}")
                    last_scan_time = time.time()
            else:
                # Chưa có kết quả, chờ thêm
                if not is_recognizing:
                    print("⚠️ Không có kết quả nhận diện.")
                    last_scan_time = time.time()
            
            # Reset trạng thái
            face_detected_time = None
            recognition_started = False
            recognition_result = None
    
    # Kiểm tra khuôn mặt mỗi 15 frames (giảm tần suất để tăng hiệu suất)
    elif frame_count % 15 == 0 and not is_recognizing:
        try:
            # Resize frame để tăng tốc
            small_frame = cv2.resize(frame, (320, 240))
            
            # Phát hiện khuôn mặt
            results = DeepFace.represent(
                img_path=small_frame, 
                model_name="Facenet512", 
                enforce_detection=True,
                detector_backend="opencv"
            )

            if results:
                # Lọc khuôn mặt trong khung oval
                display_frame, frame_center, frame_size = draw_face_guide_frame(display_frame)
                
                faces_in_frame = []
                for face in results:
                    if 'facial_area' in face:
                        if is_face_in_frame(face['facial_area'], frame_center, frame_size):
                            faces_in_frame.append(face)
                
                if faces_in_frame and face_detected_time is None:
                    # Lần đầu phát hiện khuôn mặt trong khung
                    face_detected_time = current_time
                    print("\n✓ Phát hiện khuôn mặt trong khung! Đang đếm ngược...")
                elif not faces_in_frame and face_detected_time is not None:
                    # Khuôn mặt rời khỏi khung trong quá trình countdown
                    print("⚠️ Khuôn mặt rời khỏi khung. Hủy countdown.")
                    face_detected_time = None
                    display_frame, _, _ = draw_face_guide_frame(display_frame)
                elif not faces_in_frame:
                    # Không có khuôn mặt trong khung
                    face_detected_time = None
                    display_frame, _, _ = draw_face_guide_frame(display_frame)
            else:
                # Không phát hiện khuôn mặt
                if face_detected_time is not None:
                    print("⚠️ Khuôn mặt rời khỏi khung. Hủy countdown.")
                face_detected_time = None
                display_frame, _, _ = draw_face_guide_frame(display_frame)

        except ValueError as e:
            if "Face could not be detected" in str(e):
                if face_detected_time is not None:
                    print("⚠️ Khuôn mặt rời khỏi khung. Hủy countdown.")
                face_detected_time = None
                display_frame, _, _ = draw_face_guide_frame(display_frame)
        except Exception as e:
            print(f"❌ Lỗi hệ thống: {e}")
    else:
        # Frame bình thường, chỉ vẽ khung
        if face_detected_time is None and not is_recognizing:
            display_frame, _, _ = draw_face_guide_frame(display_frame)

    # Hiển thị frame (nếu chưa được hiển thị trong countdown)
    if face_detected_time is None or is_recognizing:
        cv2.imshow("Kiosk Tra Cuu Lich Hoc FPT", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
client.close()
print("👋 Đã đóng hệ thống Kiosk.")
