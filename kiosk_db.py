import sqlite3
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
# from kiosk_config import QDRANT_PATH, SQLITE_DB, COLLECTION_NAME
QDRANT_PATH = "./qdrant_db"
SQLITE_DB = "student_info.db"
COLLECTION_NAME = "student_faces"

class DatabaseHandler:
    def __init__(self):
        # Init Qdrant
        self.client = QdrantClient(path=QDRANT_PATH)
        try:
            self.client.get_collection(COLLECTION_NAME)
        except:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
        
    def get_student_info(self, student_id):
        """Lấy thông tin sinh viên từ SQLite"""
        try:
            conn = sqlite3.connect(SQLITE_DB)
            cursor = conn.cursor()
            cursor.execute("SELECT name, schedule, room FROM students WHERE id = ?", (student_id,))
            result = cursor.fetchone()
            conn.close()
            if result:
                return result
            return ("Unknown", "N/A", "N/A")
        except Exception as e:
            print(f"DB Error: {e}")
            return ("Lỗi DB", "N/A", "N/A")

    def search_face(self, embedding):
        """Tìm kiếm vector trong Qdrant"""
        search_result = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=2
        ).points
        return search_result
