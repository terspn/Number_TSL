# src/03_test_final_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
import numpy as np

# --- 1. กำหนดค่าคงที่ ---
DATA_DIR = os.path.join('..', 'dataset') 
MODEL_PATH = os.path.join('..', 'models', 'thai_sign_numbers_cnn.h5')
TEST_DIR = os.path.join(DATA_DIR, 'testing')
IMG_SIZE = 224 # ต้องตรงกับตอนฝึกโมเดล
BATCH_SIZE = 32

# --- 2. โหลดโมเดลที่ฝึกเสร็จแล้ว ---
print(f"--- Loading Model from {MODEL_PATH} ---")
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"ERROR: Cannot load model. Please ensure 01_train_model.py has been run successfully. Error: {e}")
    exit()

# --- 3. เตรียม Testing Data Generator ---
print("--- Preparing Testing Data ---")

# สำหรับการทดสอบ (Testing) เราใช้แค่ Rescale และไม่ใช้ Data Augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False # สำคัญมาก: ห้ามสลับข้อมูลเพื่อให้ Confusion Matrix ถูกต้อง
)

# --- 4. ประเมินผลโมเดล (Evaluation) ---
print("\n--- Evaluating Model Performance ---")

# ใช้ model.evaluate เพื่อรับค่า Loss และ Accuracy
loss, accuracy = model.evaluate(
    test_generator,
    steps=test_generator.samples // BATCH_SIZE
)

print(f"\n✅ FINAL TEST ACCURACY: {accuracy * 100:.2f}%")
print(f"❌ FINAL TEST LOSS: {loss:.4f}")

# --- 5. การวิเคราะห์เชิงลึก (Confusion Matrix) ---
print("\n--- Generating Predictions for Confusion Matrix ---")

# ทำนายผลลัพธ์ของ Testing Set ทั้งหมด
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# สร้าง Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ตรวจสอบว่า sklearn ถูกติดตั้งหรือไม่ (ถ้ายังไม่ติดตั้งให้ใช้คำสั่ง: pip install scikit-learn seaborn matplotlib)
try:
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix (Testing Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    print("Confusion Matrix plot generated and shown.")

except ImportError:
    print("\nSkipping Confusion Matrix plot. Please run 'pip install scikit-learn seaborn matplotlib' for full analysis.")