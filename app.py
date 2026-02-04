from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import datetime
import os
import numpy as np
from deepface import DeepFace
from qdrant_client import QdrantClient

# Import DB handler từ module cũ
from kiosk_db import DatabaseHandler

import mediapipe as mp

import signal
import sys

app = Flask(__name__)

# --- PRELOAD MODEL (Để khởi động nhanh hơn) ---
print("🚀 Đang tải model AI...")
try:
    # Preload ArcFace model bằng cách tạo embedding giả
    import numpy as np
    dummy_img = np.zeros((112, 112, 3), dtype=np.uint8)
    DeepFace.represent(img_path=dummy_img, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
    print("✅ Model AI đã sẵn sàng!")
except:
    pass

# --- QDRANT CLIENT ---
QDRANT_PATH = "./qdrant_db"
COLLECTION_NAME = "student_faces"
# qdrant_client = QdrantClient(path=QDRANT_PATH) # REMOVED to avoid double locking

# --- GLOBAL STATE ---
class KioskState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.status = "SCANNING" # SCANNING, LIVENESS, PROCESSING, CONFIRM, SPOOF
        self.progress = 0
        self.student_data = None 
        self.last_scan_time = 0
        self.process_start_time = 0
        self.db = DatabaseHandler()
        self.running = True # Cờ kiểm soát vòng lặp
        # Liveness Blink State
        self.blink_counter = 0
        self.is_blinking = False
        self.last_blink_time = 0  # Thời gian lần chớp trước (để chống video replay)
        # Verification State
        self.consecutive_match_count = 0
        self.last_recognized_sid = None
        
state = KioskState()

# Handle Ctrl+C
def signal_handler(sig, frame):
    print('👋 Đang tắt hệ thống NGAY LẬP TỨC...')
    state.running = False
    # time.sleep(0.5)  <-- Xóa dòng này
    os._exit(0)  # Force exit ngay lập tức

signal.signal(signal.SIGINT, signal_handler)

# --- BLINK DETECTION HELPERS ---
mp_face_mesh = mp.solutions.face_mesh
# Index mắt trái/phải trong FaceMesh (Chuẩn Mediapipe)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices, w, h):
    """Tính Eye Aspect Ratio (Tỷ lệ mở mắt)"""
    # Lấy tọa độ
    coords = []
    for idx in eye_indices:
        lm = landmarks[idx]
        coords.append((lm.x * w, lm.y * h))
    
    # Tính khoảng cách dọc (Vertical)
    v1 = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    v2 = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    
    # Tính khoảng cách ngang (Horizontal)
    h_dist = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

# --- AI ENHANCEMENT HELPERS ---
def preprocess_frame(frame):
    """
    Giữ nguyên frame gốc cho AI xử lý.
    Các model hiện đại (ArcFace) hoạt động tốt nhất với dữ liệu gốc thay vì filter thủ công.
    """
    return frame

def check_img_quality(frame):
    # Tắt check chất lượng quá gắt để tránh chặn nhầm trong môi trường tối
    return True, "OK"

def check_spoofing_opencv(frame, face_area=None):
    return False, "Real"

def run_recognition_async(frame, state):
    """Chạy AI Nhận diện - Debug Mode (RGB + Low Threshold)"""
    print(f"🚀 AI Start: Kích thước ảnh {frame.shape}")
    try:
        # DeepFace làm việc tốt chuẩn với RGB
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- 1. DETECT & EXTRACT FACE ---
        print("🔍 Đang Detect Face (Mediapipe)...")
        face_objs = DeepFace.extract_faces(
            img_path=input_frame,
            detector_backend="mediapipe",
            enforce_detection=True,
            align=True,
            grayscale=False
        )
        
        if not face_objs:
            print("⚠️ DeepFace không tìm thấy mặt")
            with state.lock:
                state.status = "SCANNING"
                state.progress = 0
            return

        current_face = face_objs[0]["face"]
        print(f"✅ Face Detected. Shape: {current_face.shape}")

        if current_face.max() <= 1.0:
            current_face = (current_face * 255).astype(np.uint8)

        # --- 2. GET EMBEDDING ---
        print("🧬 Đang tạo Embedding (ArcFace)...")
        results = DeepFace.represent(
            img_path=current_face,
            model_name="ArcFace",
            detector_backend="skip",
            enforce_detection=False,
            align=True
        )
        
        if not results: 
            print("❌ Lỗi tạo Embedding")
            with state.lock:
                state.status = "SCANNING"
                state.progress = 0
            return
            
        embedding = results[0]["embedding"]
        
        # --- 3. SEARCH DATABASE ---
        print("🔎 Đang Query Qdrant...")
        search_res = state.db.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=3
        ).points
        
        found = False
        
        if search_res:
            best_match = search_res[0]
            score = best_match.score
            print(f"🎯 Top 1: {best_match.payload['student_id']} - Score: {score:.4f}")
            
            accepted_sid = None
            
            # --- 4. MATCHING LOGIC (SMART GAP CHECK) ---
            # Hạ xuống 0.40
            if score > 0.40:
                # Tìm đối thủ thực sự (người đầu tiên có ID khác)
                competitor = None
                for res in search_res[1:]:
                    if res.payload['student_id'] != best_match.payload['student_id']:
                        competitor = res
                        break
                
                if competitor:
                    gap = score - competitor.score
                    print(f"   Gap vs Different Person ({competitor.payload['student_id']}: {competitor.score:.4f}): {gap:.4f}")
                    
                    # Nếu phân vân giữa 2 người khác nhau mà khoảng cách quá hẹp (< 0.02)
                    if gap < 0.02 and score < 0.65:
                         print(f"⚠️ Từ chối: Nhập nhằng giữa {best_match.payload['student_id']} và {competitor.payload['student_id']}")
                    else:
                        accepted_sid = best_match.payload['student_id']
                else:
                    # Không có đối thủ khác ID nào trong top -> Quá an toàn
                    accepted_sid = best_match.payload['student_id']
            
            if accepted_sid:
                print(f"✅ CHẤP NHẬN MATCH: {accepted_sid}")
                
                # Bỏ qua Consecutive check để test độ nhạy -> Confirm Lập Tức
                name, sch, room = state.db.get_student_info(accepted_sid)
                with state.lock:
                    state.student_data = {
                        "name": name,
                        "student_id": accepted_sid,
                        "schedule": sch, # Map data
                        "room": room,
                        "checkin_time": datetime.datetime.now().strftime("%H:%M %d/%m")
                    }
                    state.status = "CONFIRM"
                    state.progress = 100
                    state.consecutive_match_count = 0 
                found = True
            else:
                print(f"❌ Low Score (< 0.40) hoặc Ambiguous")
        else:
            print("❌ DB Empty")
        
        if not found:
            with state.lock:
                state.status = "SCANNING"
                state.progress = 0

    except Exception as e:
        print(f"🔥 AI Exception: {e}")
        import traceback
        traceback.print_exc()
        with state.lock:
            state.status = "SCANNING"
            state.progress = 0
        return

        # QUAN TRỌNG: Nếu chưa Confirm và chưa về Scanning -> Reset timer để Camera Worker gọi tiếp
        if state.status == "PROCESSING":
            with state.lock:
                # Đặt lại thời gian để camera worker tiếp tục đếm process
                # Trừ đi 0.3s để lần sau chạy nhanh hơn (chỉ đợi 0.2s)
                state.process_start_time = time.time() - 0.3

    except Exception as e:
        print(f"AI Error: {e}")
        with state.lock:
            state.status = "SCANNING"
            state.progress = 0

# --- CAMERA THREAD ---
def camera_worker():
    cap = cv2.VideoCapture(0)
    # Giảm độ phân giải xuống 640x480 để mượt hơn
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Init Face Mesh (Vẫn dùng để detect khuôn mặt nhanh)
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while state.running:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # --- LOCK AI WHEN CONFIRMING (NEW) ---
        # Nếu đang đợi người dùng bấm nút, không làm gì cả để tiết kiệm CPU và tránh nhảy log
        if state.status == "CONFIRM":
            with state.lock:
                state.frame = frame.copy()
            time.sleep(0.1)
            continue
            
        # --- STATE MACHINE ---
        current_time = time.time()
        
        # --- MAIN FACE SELECTION (Anti-Crowd) ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            screen_center_x, screen_center_y = w // 2, h // 2
            
            best_face_data = None
            max_focus_score = -1

            # Duyệt qua tất cả mặt để tìm "Người chủ trì"
            for face_landmarks in results.multi_face_landmarks:
                # Tính bounding box
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in face_landmarks.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, cx), min(y_min, cy)
                    x_max, y_max = max(x_max, cx), max(y_max, cy)
                
                # Tính điểm ưu tiên (Diện tích * Độ trung tâm)
                area = (x_max - x_min) * (y_max - y_min)
                face_center_x = (x_min + x_max) / 2
                face_center_y = (y_min + y_max) / 2
                dist_to_center = ((face_center_x - screen_center_x)**2 + (face_center_y - screen_center_y)**2)**0.5
                
                # Heuristic: Ưu tiên mặt TO và GẦN TÂM (Trọng số diện tích cao hơn)
                focus_score = area / (dist_to_center + 1) 
                
                if focus_score > max_focus_score:
                    max_focus_score = focus_score
                    best_face_data = (x_min, y_min, x_max, y_max)

            # Chỉ vẽ và xử lý khuôn mặt TỐT NHẤT
            if best_face_data:
                x_min, y_min, x_max, y_max = best_face_data
                
                # Vẽ box (Màu Cam nếu đang Scan/Process, Màu Xanh nếu đã Confirm)
                color = (33, 111, 242) # FPT Orange
                if state.status == "CONFIRM": color = (73, 132, 30) # Green
                
                # Vẽ góc Corner
                t, l = 2, 30
                cv2.line(frame, (x_min, y_min), (x_min + l, y_min), color, t + 2)
                cv2.line(frame, (x_min, y_min), (x_min, y_min + l), color, t + 2)
                cv2.line(frame, (x_max, y_min), (x_max - l, y_min), color, t + 2)
                cv2.line(frame, (x_max, y_min), (x_max, y_min + l), color, t + 2)
                cv2.line(frame, (x_min, y_max), (x_min + l, y_max), color, t + 2)
                cv2.line(frame, (x_min, y_max), (x_min, y_max - l), color, t + 2)
                cv2.line(frame, (x_max, y_max), (x_max - l, y_max), color, t + 2)
                cv2.line(frame, (x_max, y_max), (x_max, y_max - l), color, t + 2)
                
                # Trigger Processing
                if state.status == "SCANNING" and (current_time - state.last_scan_time > 1.0):
                    with state.lock:
                        state.status = "PROCESSING"
                        state.process_start_time = current_time
                        state.progress = 0

        # 2. PROCESSING logic (Sử dụng frame đã vẽ box làm preview)
        if state.status == "PROCESSING":
            elapsed = current_time - state.process_start_time
            if elapsed < 0:
                with state.lock: state.progress = 90 + int((current_time * 10) % 9)
            else:
                prog = int((elapsed / 0.5) * 90)
                with state.lock: state.progress = min(90, max(0, prog))
                if elapsed > 0.3:
                    # Chụp frame gốc để xử lý AI
                    threading.Thread(target=run_recognition_async, args=(frame.copy(), state), daemon=True).start()
                    with state.lock: state.process_start_time = current_time + 1000 

        with state.lock:
            state.frame = frame.copy()
        
        time.sleep(0.01)

# Start Thread
t = threading.Thread(target=camera_worker, daemon=True)
t.start()

# --- WEB ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        with state.lock:
            if state.frame is None: continue
            
            # encode frame
            _, buffer = cv2.imencode('.jpg', state.frame)
            frame_bytes = buffer.tobytes()
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def get_status():
    return jsonify({
        "status": state.status,
        "progress": state.progress,
        "data": state.student_data
    })

@app.route('/api/action', methods=['POST'])
def handle_action():
    req = request.json
    action = req.get('action') # 'confirm' or 'reject'
    
    if action == 'confirm':
        # Logic Lưu điểm danh (Đã có)
        if state.student_data:
            sid = state.student_data['student_id']
            
            # --- TÍNH NĂNG TỰ HỌC (SELF-LEARNING) ---
            # Lưu ảnh xác thực vào collected_faces/MSSV/ để định kỳ training lại
            try:
                # Tạo folder riêng cho từng sinh viên
                student_collect_dir = os.path.join("collected_faces", sid)
                if not os.path.exists(student_collect_dir):
                    os.makedirs(student_collect_dir)
                
                # Format tên file: Timestamp.jpg
                filename = f"{int(time.time())}.jpg"
                save_path = os.path.join(student_collect_dir, filename)
                
                # Lưu frame tại thời điểm xác nhận
                with state.lock:
                    if state.frame is not None:
                        cv2.imwrite(save_path, state.frame)
                        print(f"📸 Đã lưu ảnh tự học vào folder: {save_path}")
            except Exception as e:
                print(f"⚠️ Lỗi lưu ảnh tự học: {e}")
            # ----------------------------------------
            
            print(f"CONFIRMED: {sid}")
    
    # Reset state ngay lập tức
    with state.lock:
        state.status = "SCANNING"
        state.student_data = None
        state.last_scan_time = time.time() + 0.5 # Delay 0.5s trước khi quét lại (Mượt hơn)
        
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
