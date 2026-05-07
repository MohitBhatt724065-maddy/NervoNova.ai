# Data preprocessing.
import os
import numpy as np
import cv2
from tqdm import tqdm

TRAIN_DIR = "/home/mohit-bhatt/Desktop/NervoNova.ai/NervoNova.ai/Datasets/archive (7)/Training"
TEST_DIR = "/home/mohit-bhatt/Desktop/NervoNova.ai/NervoNova.ai/Datasets/archive (7)/Testing"

CATEGORIES = {
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

IMG_SIZE = 224

def load_dataset(data_dir):
    images = []
    labels = []
    print(f"Processing data from: {data_dir}")
    
    for category, label in CATEGORIES.items():
        category_path = os.path.join(data_dir, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Missing folder {category_path}")
            continue
        
        for img_name in tqdm(os.listdir(category_path), desc=f"Loading {category}"):
            img_path = os.path.join(category_path, img_name)
            
            try:
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img/255.0
                images.append(img)
                labels.append(label)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


X_train, y_train = load_dataset(TRAIN_DIR)
X_test, y_test = load_dataset(TEST_DIR)

# Verifying shapes:-
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

os.makedirs("processed_data", exist_ok = True)

np.save("processed_data/X_train.npy", X_train)
np.save("processed_data/y_train.npy", y_train)
np.save("processed_data/X_test.npy", X_test)
np.save("processed_data/y_test.npy", y_test)

print("\nProcessed datasets saved successfully in 'processed_data/' folder.")

print("\n==========CLASS DISTRIBUTION ==========")

for category, label in CATEGORIES.items():
    train_count = np.sum(y_train == label)
    test_count = np.sum(y_test == label)
    print(f"{category}")
    print(f" Training: {train_count}")
    print(f" Testing: {test_count}")
    

print("\n Data preprocessing completed successfully.")
print("Your MRI dataset is now ready for model training and evaluation!")