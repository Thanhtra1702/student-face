import os
import cv2
import numpy as np
from deepface import DeepFace
from qdrant_client import QdrantClient
from pathlib import Path
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Constants
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "student_faces"
TEST_DIR = "./flipped_faces"
MODEL_NAME = "ArcFace"

client = QdrantClient(path=DB_PATH)

def preprocess_frame(frame):
    """CLAHE preprocessing"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def evaluate():
    test_images = list(Path(TEST_DIR).glob("*.jpg")) + list(Path(TEST_DIR).glob("*.png"))
    
    print(f"Bat dau danh gia tren {len(test_images)} anh...")
    print("="*80)
    
    correct = 0
    wrong = 0
    rejected = 0
    
    results_detail = []

    for img_path in test_images:
        true_id = img_path.stem.upper()
        
        frame = cv2.imread(str(img_path))
        if frame is None: 
            continue
        
        # Multi-frame averaging (3 embeddings)
        embeddings_list = []
        
        for attempt in range(3):
            enhanced = preprocess_frame(frame)
            
            # Them nhieu nhe
            if attempt > 0:
                noise = np.random.normal(0, 2, enhanced.shape).astype(np.uint8)
                enhanced = cv2.add(enhanced, noise)
            
            # Try Mediapipe first
            try:
                results = DeepFace.represent(
                    img_path=enhanced, 
                    model_name=MODEL_NAME,
                    enforce_detection=True,
                    detector_backend="mediapipe",
                    align=True
                )
            except:
                # Fallback to OpenCV
                try:
                    results = DeepFace.represent(
                        img_path=frame, 
                        model_name=MODEL_NAME,
                        enforce_detection=True,
                        detector_backend="opencv",
                        align=True
                    )
                except:
                    results = None
            
            if results and len(results) > 0:
                embeddings_list.append(results[0]["embedding"])
        
        if len(embeddings_list) == 0:
            rejected += 1
            results_detail.append((true_id, "REJECTED", 0, 0, 0))
            continue
        
        # Tinh embedding trung binh
        avg_embedding = np.mean(embeddings_list, axis=0).tolist()
        
        # Tim kiem
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=avg_embedding,
            limit=5
        ).points
        
        p_id = "REJECTED"
        score = 0
        gap = 0
        ratio = 1.0
        
        if search_result:
            best_match = search_result[0]
            score = best_match.score
            
            if len(search_result) > 1:
                gap = best_match.score - search_result[1].score
                if search_result[1].score > 0:
                    ratio = best_match.score / search_result[1].score
            
            # Re-ranking logic (same as main.py)
            accept = False
            if score > 0.60:
                accept = True
            elif score > 0.50:
                if ratio > 1.05 or gap > 0.04:
                    accept = True
            elif score > 0.40:
                if ratio > 1.10 and gap > 0.06:
                    accept = True
            elif score > 0.35:
                if ratio > 1.15 and gap > 0.10:
                    accept = True
            
            if accept:
                p_id = best_match.payload["student_id"].upper()
        
        if p_id.upper() == true_id.upper():
            correct += 1
            status = "Correct"
        elif p_id == "REJECTED":
            rejected += 1
            status = "Rejected"
        else:
            wrong += 1
            status = "Wrong"
        
        results_detail.append((true_id, p_id, score, gap, ratio))
        print(f"{img_path.name:<20} | True: {true_id:<12} | Pred: {p_id:<12} | Score: {score:.3f} | Gap: {gap:.3f} | Ratio: {ratio:.2f} | {status}")
    
    print("="*80)
    print(f"\nTONG KET:")
    print(f"  Tong so anh:     {len(test_images)}")
    print(f"  Nhan dien dung:  {correct} ({(correct/len(test_images)*100):.2f}%)")
    print(f"  Nhan dien sai:   {wrong} ({(wrong/len(test_images)*100):.2f}%)")
    print(f"  Tu choi:         {rejected} ({(rejected/len(test_images)*100):.2f}%)")
    print(f"\n=> DO CHINH XAC: {(correct/len(test_images)*100):.2f}%")

if __name__ == "__main__":
    evaluate()
