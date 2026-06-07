import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import sqlite3
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import cosine

# Load model
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Kết nối DB
conn = sqlite3.connect('attendance.db')  # hoặc faces.db
# Giả sử bạn có hàm get_all_embeddings()

def get_embedding(image):
    faces = app.get(image)
    if len(faces) == 0:
        return None
    return faces[0].normed_embedding  # 512-dim

# ================== EVALUATION ==================
def evaluate_model(test_images, ground_truth_labels, threshold=0.6):
    y_true = []
    y_pred = []
    similarities = []
    
    for img_path, true_label in zip(test_images, ground_truth_labels):
        img = cv2.imread(img_path)
        emb = get_embedding(img)
        
        if emb is None:
            y_pred.append("Unknown")
            y_true.append(true_label)
            continue
            
        # So sánh với tất cả trong DB
        best_score = -1
        best_match = "Unknown"
        
        for name, db_emb in registered_embeddings.items():  # từ DB
            score = 1 - cosine(emb, db_emb)
            if score > best_score:
                best_score = score
                best_match = name
                
        predicted = best_match if best_score >= threshold else "Unknown"
        
        y_true.append(true_label)
        y_pred.append(predicted)
        similarities.append(best_score)
    
    # Tính metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # FAR & FRR (cần tính riêng impostor/genuine pairs)
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-Score : {f1*100:.2f}%")
    print(f"Threshold: {threshold}")
    
    return acc, prec, rec, f1