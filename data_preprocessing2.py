# Data preproccing using memory efficient method.
import os
import numpy as np
import cv2
from tqdm import tqdm

TRAIN_DIR = "/home/mohit-bhatt/Desktop/NervoNova.ai/NervoNova.ai/Datasets/archive (7)/Training"
TEST_DIR = "/home/mohit-bhatt/Desktop/NervoNova.ai/NervoNova.ai/Datasets/archive (7)/Testing"

CATEGORIES ={
    "glioma": 0,
    "meningioma": 1,
    "notumor": 2,
    "pituitary": 3
}

IMG_SIZE = 224
os.makedirs("Processed_Data", exist_ok = True)

def count_images(data_dir):
    total_images = 0
    for category in CATEGORIES.keys():
        category_path = os.path.join(data_dir, category)
        if os.path.exists(category_path):
            total_images += len([f for f in os.listdir(category_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(category_path)
            print(total_images)
            
    return total_images

def load_dataset_efficient(data_dir, save_prefix):
    print(f"\nProcessing: {data_dir}")
    
    
    total = count_images(data_dir)
    print(f"Total images found: {total}")
    
    X = np.memmap(f"Processed_Data/{save_prefix}_X.dat",
                  dtype=np.float32, mode='w+',
                  shape=(total, IMG_SIZE, IMG_SIZE, 3))
    y = np.memmap(f"Processed_Data/{save_prefix}_y.dat",
                  dtype=np.int32, mode='w+',
                  shape=(total,))
    
    idx = 0  
    
    
    for category, label in CATEGORIES.items():
        category_path = os.path.join(data_dir, category)
        
        if not os.path.exists(category_path):
            print(f"Warning: Missing folder {category_path}")
            continue
        
        image_files = [f for f in os.listdir(category_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(image_files, desc=f"Loading {category}"):
            img_path = os.path.join(category_path, img_name)
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # normalize
                
                
                X[idx] = img
                y[idx] = label
                idx += 1
                
            except Exception as e:
                print(f"Error: {img_path}: {e}")
    
   
    X.flush()  
    y.flush()
    
    
    np.save(f"Processed_Data/{save_prefix}_shape.npy", np.array([idx, IMG_SIZE, IMG_SIZE, 3]))
    
    print(f"Saved {idx} images to disk.")
    return idx

train_count = load_dataset_efficient(TRAIN_DIR, "train")
test_count  = load_dataset_efficient(TEST_DIR, "test")



print("\n✅ Preprocessing complete! Data saved to processed_data/ folder.")

import numpy as np

# See the shape (how many images, size)
shape = np.load("Processed_Data/train_shape.npy")
print(shape)  # e.g., [5600, 224, 224, 3]

# View a single image
import matplotlib.pyplot as plt
train_X = np.memmap("Processed_Data/train_X.dat", dtype='float32', mode='r', shape=tuple(shape))
plt.imshow(train_X[0])  # shows first image
plt.show()