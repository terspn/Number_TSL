# src/02_realtime_app.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# --- 1. กำหนดค่าเริ่มต้น (Configuration) ---

MODEL_PATH = os.path.join('..', 'models', 'thai_sign_numbers_cnn.h5') 

# โหลดโมเดล CNN ที่คุณฝึกไว้สำหรับตัวเลข
try:
    model = tf.keras.models.load_model(MODEL_PATH) 
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_PATH}' exists and is a valid Keras model (รัน 01_train_model.py ก่อน).")
    exit()

# กำหนดขนาดรูปภาพที่โมเดล CNN คาดหวัง (ต้องตรงกับตอนฝึก)
IMG_SIZE = 224 
# กำหนดป้ายชื่อ (Labels) ตามลำดับคลาสของโมเดล (สำคัญ: ต้องเรียงตามชื่อโฟลเดอร์ 0-9)
LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# กำหนดค่า MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- 2. เปิดกล้อง (Initialize Camera) ---
# ใช้กล้องหลัก (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้ โปรดตรวจสอบว่ากล้องถูกใช้งานอยู่หรือไม่")
    exit()

print("ระบบพร้อมทำงาน: แสดงภาษามือตัวเลขหน้ากล้อง...")

# --- 3. วนลูปประมวลผลวิดีโอ (Main Loop) ---

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # พลิกภาพ (Mirror effect)
    frame = cv2.flip(frame, 1)
    
    # แปลงสีเป็น RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # ประมวลผลด้วย MediaPipe Hands
    results = hands.process(rgb_frame)
    
    predicted_label = "Waiting for hand..."
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # วาดจุด Landamarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # --- 3.1 การครอบตัดภาพ (Hand Cropping) ---
            
            h, w, c = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            
            # หา Bounding Box
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # ขยายขอบเขต (Padding)
            padding = 30
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            cropped_hand = frame[y_min:y_max, x_min:x_max]
            
            if cropped_hand.size > 0:
                # --- 3.2 การเตรียมภาพและทำนายผล (Prediction) ---
                
                # ปรับขนาดภาพ
                try:
                    input_img = cv2.resize(cropped_hand, (IMG_SIZE, IMG_SIZE))
                except cv2.error:
                    continue 
                
                # แปลง Array และ Normalize (0-1)
                input_img = np.expand_dims(input_img, axis=0) / 255.0
                
                # ทำนายผล
                predictions = model.predict(input_img, verbose=0)
                predicted_class = np.argmax(predictions)
                
                # ดึงป้ายชื่อ
                predicted_label = LABELS[predicted_class]
                confidence = predictions[0][predicted_class] * 100
                
                # --- 3.3 แสดงผลบนหน้าจอ ---
                
                # วาดสี่เหลี่ยมรอบมือ
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # แสดงผลลัพธ์การทำนาย
                result_text = f"Result: {predicted_label} ({confidence:.2f}%)"
                cv2.putText(frame, result_text, (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            break 
            
    # แสดงภาพวิดีโอ
    cv2.imshow('Thai Sign Language Number Recognition', frame)
    
    # กด 'q' เพื่อออก
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- 4. ปิดกล้องและหน้าต่าง (Cleanup) ---
cap.release()
cv2.destroyAllWindows()
hands.close()