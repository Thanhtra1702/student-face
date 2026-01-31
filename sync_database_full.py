import sqlite3
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

def sync_database():
    """Đồng bộ student_info.db với ảnh trong database/"""
    conn = sqlite3.connect('student_info.db')
    cursor = conn.cursor()
    
    # 1. Lấy danh sách ảnh hiện có
    image_dir = "./database"
    if not os.path.exists(image_dir):
        print("Không tìm thấy thư mục database!")
        return
    
    files = os.listdir(image_dir)
    student_ids_from_images = set()
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            s_id = os.path.splitext(f)[0]
            student_ids_from_images.add(s_id)
    
    print(f"Tìm thấy {len(student_ids_from_images)} file ảnh trong database/")
    
    # 2. Lấy danh sách sinh viên trong DB
    cursor.execute("SELECT id FROM students")
    student_ids_in_db = set([row[0] for row in cursor.fetchall()])
    
    print(f"Có {len(student_ids_in_db)} sinh viên trong student_info.db")
    
    # 3. Tìm sinh viên cần thêm (có ảnh nhưng chưa có trong DB)
    to_add = student_ids_from_images - student_ids_in_db
    
    # 4. Tìm sinh viên cần xóa (có trong DB nhưng không có ảnh)
    to_remove = student_ids_in_db - student_ids_from_images
    
    print("\n" + "="*80)
    
    # 5. Thêm sinh viên mới
    if to_add:
        print(f"\n➕ Thêm {len(to_add)} sinh viên mới:")
        for s_id in sorted(to_add):
            cursor.execute('''
                INSERT INTO students (id, name, schedule, room) 
                VALUES (?, ?, ?, ?)
            ''', (s_id, f"Sinh viên {s_id}", "DPL302m (12:30-14:45)", "305"))
            print(f"  + {s_id}")
    else:
        print("\n✓ Không có sinh viên mới cần thêm")
    
    # 6. Xóa sinh viên không còn ảnh
    if to_remove:
        print(f"\n➖ Xóa {len(to_remove)} sinh viên không còn ảnh:")
        for s_id in sorted(to_remove):
            cursor.execute("DELETE FROM students WHERE id = ?", (s_id,))
            print(f"  - {s_id}")
    else:
        print("\n✓ Không có sinh viên cần xóa")
    
    conn.commit()
    
    # 7. Kiểm tra lại
    cursor.execute("SELECT COUNT(*) FROM students")
    final_count = cursor.fetchone()[0]
    
    conn.close()
    
    print("\n" + "="*80)
    print(f"\n✅ Đồng bộ hoàn tất!")
    print(f"📊 Tổng số sinh viên trong DB: {final_count}")
    print(f"📊 Tổng số ảnh trong database/: {len(student_ids_from_images)}")
    
    if final_count == len(student_ids_from_images):
        print("✓ Database đã đồng bộ hoàn toàn!")
    else:
        print("⚠️ Vẫn có sự khác biệt, cần kiểm tra lại!")

if __name__ == "__main__":
    sync_database()
