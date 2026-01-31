"""
Script để tạo database SQLite mẫu với thông tin sinh viên
Tự động quét thư mục ./database và thêm sinh viên mới nếu chưa có trong danh sách cứng
"""

import sqlite3
import sys
import os
import random

# Đảm bảo in được tiếng Việt trên môi trường Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

DATABASE_DIR = "./database"

def create_database():
    # Kết nối (hoặc tạo mới) database
    conn = sqlite3.connect('student_info.db')
    cursor = conn.cursor()
    
    # Tạo bảng students
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            schedule TEXT,
            room TEXT
        )
    ''')
    
    # Dữ liệu mẫu ban đầu (Hardcoded)
    sample_students = {
        'QE190002': {'name': 'Thân Phúc Hậu', 'schedule': 'TMG301 (12:30-14:45)', 'room': '314'},
        'QE190012': {'name': 'Nguyễn Châu Thành Sơn', 'schedule': 'SEG301 (12:30-14:45)', 'room': '318'},
        'QE190023': {'name': 'Phan Minh Tài', 'schedule': 'DPL302m (12:30-14:45)', 'room': '305'},
        'QE190027': {'name': 'Nguyễn Hải Nam', 'schedule': 'TMG301 (12:30-14:45)', 'room': '314'},
        'QE190029': {'name': 'Võ Quang Trường', 'schedule': 'SEG301 (12:30-14:45)', 'room': '318'},
        'QE190039': {'name': 'Tô Thanh Hậu', 'schedule': 'DPL302m (12:30-14:45)', 'room': '305'},
        'QE190047': {'name': 'Phạm Quang Chiến', 'schedule': 'TMG301 (12:30-14:45)', 'room': '314'},
        'QE190057': {'name': 'Nguyễn Nhất Long', 'schedule': 'SEG301 (12:30-14:45)', 'room': '318'},
        'QE190059': {'name': 'Võ Minh Huy', 'schedule': 'DPL302m (12:30-14:45)', 'room': '305'},
        'QE190076': {'name': 'Ngô Tuấn Hoàng', 'schedule': 'TMG301 (12:30-14:45)', 'room': '314'},
        'QE190099': {'name': 'Nguyễn Thanh Trà', 'schedule': 'SEG301 (12:30-14:45)', 'room': '318'},
        'QE190109': {'name': 'Châu Thái Nhật Minh', 'schedule': 'DPL302m (12:30-14:45)', 'room': '305'},
        'QE190123': {'name': 'Phan Đỗ Thanh Tuấn', 'schedule': 'TMG301 (12:30-14:45)', 'room': '314'},
        'QE190129': {'name': 'Trịnh Khải Nguyên', 'schedule': 'SEG301 (12:30-14:45)', 'room': '318'},
        'QE190134': {'name': 'Nguyễn Lê Anh Duy', 'schedule': 'DPL302m (12:30-14:45)', 'room': '305'},
        'QE190136': {'name': 'Đặng Văn Hậu', 'schedule': 'TMG301 (12:30-14:45)', 'room': '314'},
        'QE190142': {'name': 'Lê Hoàng Hữu', 'schedule': 'SEG301 (12:30-14:45)', 'room': '318'},
        'QE190155': {'name': 'Nguyễn Lê Tấn Pháp', 'schedule': 'DPL302m (12:30-14:45)', 'room': '305'},
        'QE190162': {'name': 'Trần Gia Phúc', 'schedule': 'TMG301 (12:30-14:45)', 'room': '314'},
    }
    
    # Các templates lịch học để random cho sinh viên mới
    schedule_templates = [
        {'schedule': 'TMG301 (12:30-14:45)', 'room': '314'},
        {'schedule': 'SEG301 (12:30-14:45)', 'room': '318'},
        {'schedule': 'DPL302m (12:30-14:45)', 'room': '305'},
        {'schedule': 'IOT102 (07:30-09:45)', 'room': '205'},
        {'schedule': 'MAD101 (10:00-12:15)', 'room': '401'}
    ]

    # Quét thư mục ảnh
    print(f"Đang quét thư mục: {DATABASE_DIR}")
    if os.path.exists(DATABASE_DIR):
        files = os.listdir(DATABASE_DIR)
        image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        count_new = 0
        for img_file in image_files:
            # Lấy tên file làm ID (bỏ đuôi mở rộng)
            student_id = os.path.splitext(img_file)[0]
            
            # Nếu ID chưa có trong danh sách mẫu, thêm vào với dữ liệu random
            if student_id not in sample_students:
                template = random.choice(schedule_templates)
                # Xử lý tên hiển thị cho đẹp một chút (Tùy chọn)
                display_name = f"Sinh viên {student_id}"
                
                sample_students[student_id] = {
                    'name': display_name,
                    'schedule': template['schedule'],
                    'room': template['room']
                }
                count_new += 1
                
        print(f"✅ Đã tìm thấy {len(image_files)} ảnh. Thêm mới {count_new} sinh viên vào danh sách.")
    else:
        print(f"⚠️ Thư mục {DATABASE_DIR} không tồn tại!")

    # Chuyển đổi sang list các tuple để insert
    data_to_insert = []
    for sid, info in sample_students.items():
        data_to_insert.append((sid, info['name'], info['schedule'], info['room']))

    # Xóa dữ liệu cũ và thêm dữ liệu mới
    cursor.execute('DELETE FROM students')
    cursor.executemany('INSERT INTO students (id, name, schedule, room) VALUES (?, ?, ?, ?)', data_to_insert)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Đã cập nhật database student_info.db với {len(data_to_insert)} sinh viên!")
    # print("\nDanh sách sinh viên (Top 20):")
    # print("-" * 60)
    # for i, student in enumerate(data_to_insert[:20]):
    #     print(f"MSSV: {student[0]} | Tên: {student[1]}")
    # print("-" * 60)

if __name__ == "__main__":
    create_database()
