import sqlite3
import os
import sys

# Đảm bảo in được tiếng Việt trên Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def update_dummy_data():
    conn = sqlite3.connect('student_info.db')
    cursor = conn.cursor()
    
    # Lấy danh sách ảnh hiện có (đây là nguồn Source of Truth)
    # Tên file ảnh = Student ID (ví dụ: QE12345.jpg -> id=QE12345)
    image_dir = "./database"
    if not os.path.exists(image_dir):
        print("Không tìm thấy thư mục database!")
        return

    files = os.listdir(image_dir)
    student_ids_from_images = []
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Lấy tên file làm ID (bỏ đuôi mở rộng)
            s_id = os.path.splitext(f)[0]
            student_ids_from_images.append(s_id)

    print(f"Tìm thấy {len(student_ids_from_images)} file ảnh.")

    count_new = 0
    for s_id in student_ids_from_images:
        # Kiểm tra xem ID này đã có trong DB chưa
        cursor.execute("SELECT id FROM students WHERE id = ?", (s_id,))
        exists = cursor.fetchone()
        
        if not exists:
            # Nếu chưa có, insert dummy data
            cursor.execute('''
                INSERT INTO students (id, name, schedule, room) 
                VALUES (?, ?, ?, ?)
            ''', (s_id, f"Sinh viên {s_id}", "Lịch học giả định (Test)", "P.Test"))
            count_new += 1
            
    conn.commit()
    conn.close()
    
    print(f"✅ Đã cập nhật xong!")
    print(f"➕ Đã thêm mới {count_new} sinh viên vào database.")
    print(f"ℹ️ Tổng số sinh viên hiện có khớp với số lượng ảnh.")

if __name__ == "__main__":
    update_dummy_data()
