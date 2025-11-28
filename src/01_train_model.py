# src/01_train_model.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os

# --- 1. กำหนดค่าคงที่ ---
DATA_DIR = os.path.join('..', 'dataset') 
MODEL_SAVE_PATH = os.path.join('..', 'models', 'thai_sign_numbers_cnn.h5')
IMG_SIZE = 224      
BATCH_SIZE = 32
NUM_CLASSES = 10    
INITIAL_EPOCHS = 25     # รอบการฝึกฝน Stage 1
FINE_TUNE_EPOCHS = 20  # รอบการฝึกฝน Stage 2
FINE_TUNE_LR = 5e-6     # Learning Rate ที่ต่ำมากสำหรับการ Fine-tuning

# --- 2. เตรียมข้อมูล (Data Augmentation & Loading) ---
print("--- Preparing Data ---")

# Data Augmentation และ Normalization (1/255)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'training'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'validation'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# --- 3. สร้างโมเดล (Stage 1: Classification Head Training) ---
print("--- Stage 1: Building Model (Freeze Base) ---")

# โหลด MobileNetV2 โดยไม่รวมส่วนหัว และตรึงน้ำหนักไว้
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

for layer in base_model.layers:
    layer.trainable = False

# เพิ่มส่วนหัวใหม่สำหรับ Classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.25)(x) # เพิ่ม Dropout เพื่อลด Overfitting (New)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# --- 4. ฝึกฝน Stage 1 ---
print(f"--- Stage 1: Training Classification Head for {INITIAL_EPOCHS} epochs ---")

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=INITIAL_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 5. Stage 2: Fine-tuning (Unfreeze Layers) ---
print("\n--- Stage 2: Fine-tuning Base Model ---")

# 5.1 ปลดล็อกฐานโมเดล
# เราจะปลดล็อก MobileNetV2 บางส่วน (เช่น 50 ชั้นสุดท้าย)
# เนื่องจาก MobileNetV2 มีทั้งหมด 155 ชั้น เราจะเริ่ม Train ตั้งแต่ชั้นที่ 105
base_model.trainable = True
for layer in base_model.layers[:-50]: # ตรึง 105 ชั้นแรกไว้
    layer.trainable = False

# 5.2 คอมไพล์ใหม่ด้วย Learning Rate ที่ต่ำมาก
model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR), # ใช้อัตราการเรียนรู้ที่ต่ำกว่า
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"--- Stage 2: Fine-tuning the last 50 layers for {FINE_TUNE_EPOCHS} epochs ---")

# 5.3 ฝึกฝนต่อ
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)


# --- 6. บันทึกโมเดล ---
print(f"\n--- Saving Final Model to {MODEL_SAVE_PATH} ---")
os.makedirs(os.path.join('..', 'models'), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print("Training complete. Model saved.")