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

import signal
import sys

app = Flask(__name__)

# --- GLOBAL STATE ---
class KioskState:
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()
        self.status = "SCANNING" # SCANNING, PROCESSING, CONFIRM
        self.progress = 0
        self.student_data = None 
        self.last_scan_time = 0
        self.process_start_time = 0
        self.db = DatabaseHandler()
        self.running = True # Cờ kiểm soát vòng lặp
        
state = KioskState()

# Handle Ctrl+C
def signal_handler(sig, frame):
    print('👋 Đang tắt hệ thống...')
    state.running = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def run_recognition_async(frame):
    """Chạy AI trong thread riêng để không block camera"""
    try:
        # DeepFace Processing
        results = DeepFace.represent(frame, model_name="ArcFace", detector_backend="opencv", enforce_detection=False)
        
        found = False
        if results:
            target_embedding = results[0]["embedding"]
            # Search DB
            search_res = state.db.search_face(target_embedding)
            
            if search_res and search_res[0].score > 0.45:
                match = search_res[0]
                sid = match.payload['student_id']
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
                found = True

        if not found:
            # Không nhận ra ai
            with state.lock:
                state.status = "SCANNING"
                state.progress = 0
                state.last_scan_time = time.time() + 1.0 # Delay nhẹ

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
    # Init Face Detector (Haar) cho nhanh để trigger
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while state.running:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # --- STATE MACHINE ---
        current_time = time.time()
        
        # 1. SCANNING: Tìm mặt
        if state.status == "SCANNING" and (current_time - state.last_scan_time > 1.0):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                with state.lock:
                    state.status = "PROCESSING"
                    state.process_start_time = current_time
                    state.progress = 0

        # 2. PROCESSING: Giả lập Loading -> Trigger AI
        elif state.status == "PROCESSING":
            elapsed = current_time - state.process_start_time
            
            if elapsed < 0:
                # Đang đợi AI (do đã set start_time = tương lai)
                # Giữ progress ở mức 91-99% cho sinh động
                with state.lock:
                    state.progress = 90 + int((current_time * 10) % 9)
            else:
                # Giai đoạn loading ban đầu (0-90%)
                prog = int((elapsed / 0.8) * 90)
                
                with state.lock:
                    state.progress = min(90, max(0, prog)) # Chặn số âm
                
                if elapsed > 0.8:
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
        # Logic Lưu điểm danh ở đây
        if state.student_data:
            sid = state.student_data['student_id']
            # Save image logic...
            print(f"CONFIRMED: {sid}")
    
    # Reset state
    with state.lock:
        state.status = "SCANNING"
        state.student_data = None
        state.last_scan_time = time.time() + 2.0 # Delay 2s trước khi quét lại
        
    return jsonify({"success": True})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
