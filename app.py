from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import time
import datetime
import os
import numpy as np
from deepface import DeepFace

# Import DB handler từ module cũ
from kiosk_db import DatabaseHandler

import mediapipe as mp

import signal
import sys

app = Flask(__name__)

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
        # Verification State
        self.consecutive_match_count = 0
        self.last_recognized_sid = None
        
state = KioskState()

# Handle Ctrl+C
def signal_handler(sig, frame):
    print('👋 Đang tắt hệ thống...')
    state.running = False
    time.sleep(0.5)  # Cho camera thread kịp dừng
    os._exit(0)  # Force exit để không bị treo

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

def check_img_quality(frame):
    """Kiểm tra ảnh có bị mờ hoặc quá tối không"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Check độ mờ (Blur) - Variance of Laplacian
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 60: # Ngưỡng mờ (càng thấp càng mờ)
        return False, "Ảnh quá mờ"
        
    # 2. Check độ sáng
    avg_brightness = np.mean(gray)
    if avg_brightness < 30: return False, "Quá tối"
    if avg_brightness > 220: return False, "Quá sáng"
    
    return True, "OK"

def check_spoofing_opencv(frame):
    """
    Check giả mạo - Tối ưu hiệu năng bằng cách resize
    """
    try:
        small_frame = cv2.resize(frame, (640, 360))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Phát hiện viền hình chữ nhật
        edges = cv2.Canny(gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                area = cv2.contourArea(cnt)
                if area > 5000: 
                    return True, "Phát hiện khung thiết bị"

        # 2. Check độ lóa
        _, max_val, _, _ = cv2.minMaxLoc(gray)
        if max_val >= 253:
            return True, "Ánh sáng quá chói"
            
        return False, "Real"
    except:
        return False, "Error"

def run_recognition_async(frame):
    """Chạy AI Nhận diện (Sau khi đã qua Liveness Check)"""
    try:
        # Tiền xử lý
        enhanced_frame = preprocess_frame(frame)

        # DeepFace Processing
        try:
            results = DeepFace.represent(
                img_path=enhanced_frame, 
                model_name="ArcFace", 
                detector_backend="mediapipe",
                enforce_detection=True,
                align=True
            )
        except:
             # Fallback nếu lỗi hoặc không tìm thấy mặt
             with state.lock:
                state.status = "SCANNING"
                state.progress = 0
             return

        found = False
        if results:
            target_embedding = results[0]["embedding"]
            search_res = state.db.search_face(target_embedding)
            
            # Hạ ngưỡng xuống 0.55 (Dễ nhận diện hơn)
            # An toàn nhờ cơ chế Double Check 3 lần
            if search_res and search_res[0].score > 0.55:
                match = search_res[0]
                sid = match.payload['student_id']
                
                # --- LOGIC XÁC NHẬN KÉP (DOUBLE CHECK) ---
                if sid == state.last_recognized_sid:
                     state.consecutive_match_count += 1
                else:
                     state.consecutive_match_count = 1 # Reset nếu đổi người
                     state.last_recognized_sid = sid
                
                print(f"👀 Nhận diện: {sid} (Score: {match.score:.2f}) | Count: {state.consecutive_match_count}/3")

                # Chỉ Confirm nếu nhận đúng 3 lần liên tiếp
                if state.consecutive_match_count >= 3:
                    name, sch, room = state.db.get_student_info(sid)
                    
                    with state.lock:
                        state.student_data = {
                            "name": name,
                            "student_id": sid,
                            "schedule": sch,
                            "room": room,
                            "checkin_time": datetime.datetime.now().strftime("%H:%M %d/%m")
                        }
                        state.status = "CONFIRM"
                        state.progress = 100
                        # Reset counter
                        state.consecutive_match_count = 0
                        state.last_recognized_sid = None
                    found = True
                else:
                    # Vẫn đang trong quá trình verify -> coi như chưa found để loop tiếp
                     found = False 
            else:
                 # Score thấp -> Reset counter
                 state.consecutive_match_count = 0
                 state.last_recognized_sid = None

        if not found:
            # Không nhận ra ai hoặc đang verify
            if state.consecutive_match_count > 0:
                 pass 
            else:
                with state.lock:
                    state.status = "SCANNING"
                    state.progress = 0
                    state.last_scan_time = time.time() + 1.0 

    except Exception as e:
        print(f"AI Error: {e}")
        with state.lock:
            state.status = "SCANNING"
            state.progress = 0

# --- CAMERA THREAD ---
def camera_worker():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Init Face Mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False, # Tắt để nhanh hơn
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    frame_count = 0
    while state.running:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1
        
        # --- STATE MACHINE ---
        current_time = time.time()
        
        # 1. SCANNING -> Chuyển sang Check Liveness nếu thấy mặt (mỗi 5 frame)
        if state.status == "SCANNING" and (frame_count % 5 == 0):
            # Resize để xử lý cực nhanh
            small_rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(small_rgb)
            
            if results.multi_face_landmarks:
                with state.lock:
                    state.status = "LIVENESS"
                    state.blink_counter = 0
                    state.is_blinking = False

        # 2. LIVENESS CHECK -> Chỉ quét mỗi 2 frame
        elif state.status == "LIVENESS" and (frame_count % 2 == 0):
            # -- Check Spoofing --
            is_spoof, msg = check_spoofing_opencv(frame)
            if is_spoof:
                with state.lock:
                    state.status = "SPOOF"
                time.sleep(1.0)
                with state.lock:
                    if state.status == "SPOOF": state.status = "SCANNING"
                continue

            # Resize ảnh AI Input
            small_rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(small_rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                # Tọa độ landmark là 0-1 nên dùng w, h nào cũng được tỉ lệ đúng
                left_ear = calculate_ear(landmarks, LEFT_EYE, 640, 360)
                right_ear = calculate_ear(landmarks, RIGHT_EYE, 640, 360)
                avg_ear = (left_ear + right_ear) / 2.0
                
                if avg_ear < 0.22: # Nhạy hơn xíu
                    state.is_blinking = True
                
                if avg_ear > 0.30 and state.is_blinking:
                    with state.lock:
                        state.blink_counter += 1
                        state.is_blinking = False
                        state.status = "PROCESSING"
                        state.process_start_time = current_time
                        state.progress = 0
            else:
                with state.lock:
                    state.status = "SCANNING"

        # 3. PROCESSING (Giữ nguyên logic cũ)
        elif state.status == "PROCESSING":
            elapsed = current_time - state.process_start_time
            
            if elapsed < 0:
                # Đang đợi AI (do đã set start_time = tương lai)
                # Giữ progress ở mức 91-99% cho sinh động
                with state.lock:
                    state.progress = 90 + int((current_time * 10) % 9)
            else:
                # Giai đoạn loading ban đầu (0-90%)
                prog = int((elapsed / 0.5) * 90) # Nhanh hơn chút
                
                with state.lock:
                    state.progress = min(90, max(0, prog)) # Chặn số âm
                
                if elapsed > 0.5:
                    # Đủ thời gian chờ -> Chạy AI Thread
                    threading.Thread(target=run_recognition_async, args=(frame.copy(),), daemon=True).start()
                    # Đánh dấu đã chạy bằng cách đẩy thời gian về tương lai
                    with state.lock:
                        state.process_start_time = current_time + 1000 

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
    
    # Reset state
    with state.lock:
        state.status = "SCANNING"
        state.student_data = None
        state.last_scan_time = time.time() + 2.0 # Delay 2s trước khi quét lại
        
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
