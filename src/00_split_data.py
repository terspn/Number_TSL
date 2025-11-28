# src/00_split_data.py

import os
import random
import shutil

# --- 1. กำหนดค่าคงที่ ---
# โฟลเดอร์ที่เก็บข้อมูลดิบทั้งหมด (source)
SOURCE_DATA_DIR = '../dataset/numbers/' 
# โฟลเดอร์ปลายทางสำหรับการฝึก (80%)
TRAIN_DIR = '../dataset/training/'
# โฟลเดอร์ปลายทางสำหรับการตรวจสอบ (10%)
VALIDATION_DIR = '../dataset/validation/'
# โฟลเดอร์ปลายทางสำหรับการทดสอบ (10%)
TEST_DIR = '../dataset/testing/'

# อัตราส่วนการแบ่งข้อมูล
TRAIN_RATIO = 0.8
VALIDATION_RATIO = 0.1
TEST_RATIO = 0.1

# ตรวจสอบว่ารวมกันได้ 100%
assert TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO == 1.0

# --- 2. ฟังก์ชันหลักในการแบ่งข้อมูล ---
def split_data(source_dir, train_dir, validation_dir, test_dir, train_ratio, val_ratio, test_ratio):
    print("Starting data split...")
    
    # 1. วนลูปผ่านแต่ละคลาส (0, 1, 2, ..., 9)
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        
        # ตรวจสอบว่าเป็นโฟลเดอร์คลาสจริงๆ
        if not os.path.isdir(class_path):
            continue

        # 2. อ่านรายการไฟล์ทั้งหมดในคลาสนั้น
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"Skipping empty class: {class_name}")
            continue

        # สุ่มลำดับไฟล์
        random.shuffle(files)
        
        total_files = len(files)
        
        # 3. คำนวณจำนวนไฟล์สำหรับแต่ละส่วน
        train_count = int(total_files * train_ratio)
        validation_count = int(total_files * val_ratio)
        # ที่เหลือทั้งหมดคือ test (เพื่อหลีกเลี่ยง error จากการปัดเศษ)
        test_count = total_files - train_count - validation_count 

        # 4. แบ่งรายการไฟล์ตามจำนวนที่คำนวณไว้
        train_files = files[:train_count]
        validation_files = files[train_count : train_count + validation_count]
        test_files = files[train_count + validation_count :]

        print(f"\nClass {class_name}: Total={total_files}, Train={train_count}, Val={validation_count}, Test={test_count}")

        # 5. สร้างโฟลเดอร์คลาสในปลายทางถ้ายังไม่มี
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # 6. ย้ายไฟล์ไปยังโฟลเดอร์ปลายทางที่ถูกต้อง
        def move_files(file_list, destination_dir):
            for filename in file_list:
                src_path = os.path.join(class_path, filename)
                dst_path = os.path.join(destination_dir, class_name, filename)
                # ใช้ shutil.move ในการย้ายไฟล์ (ถ้าใช้ shutil.copy จะเป็นการคัดลอก)
                shutil.move(src_path, dst_path)
                
        move_files(train_files, train_dir)
        move_files(validation_files, validation_dir)
        move_files(test_files, test_dir)
        
    print("\nData splitting complete!")
    print("All files have been moved from 'dataset/numbers' to 'training', 'validation', and 'testing'.")
    print("You can now delete the empty 'dataset/numbers' folder.")

# --- 3. รันสคริปต์ ---
if __name__ == "__main__":
    split_data(SOURCE_DATA_DIR, TRAIN_DIR, VALIDATION_DIR, TEST_DIR, 
               TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO)