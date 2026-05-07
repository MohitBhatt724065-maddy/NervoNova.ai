# Streaming/Batch loading strategy.
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2

class BrainTumorDataset(Dataset):
    def __init__(self, data_prefix):
        shape = np.load(f"Processed_Data/{data_prefix}_shape.npy")
        self.total = shape[0]
        self.IMG_SIZE = shape[1]
        
        self.X = np.memmap(f"Processed_Data/{data_prefix}_X.dat", dtype='float32', mode='r', shape=(self.total, self.IMG_SIZE, self.IMG_SIZE, 3))
        self.y = np.memmap(f"Processed_Data/{data_prefix}_y.dat", dtype='int32', mode='r', shape=(self.total,))
        self.class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        print(f"Dataset ready: {self.total} images loaded from {data_prefix}")
        
   

    def __len__(self):
        return self.total 

    def __getitem__(self, idx):
        label = self.y[idx]
        img = self.X[idx]
        return img, label


print("Setting up datasets...")
train_dataset = BrainTumorDataset("train")
test_dataset = BrainTumorDataset("test")


print(f"Total training samples: {len(train_dataset)}")
print(f"Total testing samples: {len(test_dataset)}")

print("Creating Dataloaders...")
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"batch size is:- {BATCH_SIZE}")
print(f"Total training batches: {len(train_loader)}")
print(f"Total testing batches: {len(test_loader)}")


print("\nLoading one batch to verify...")

# Get first batch
images_batch, labels_batch = next(iter(train_loader))

print(f"Batch image shape : {images_batch.shape}")
# Expected: (32, 224, 224, 3) → 32 images, 224x224, RGB

print(f"Batch label shape : {labels_batch.shape}")
# Expected: (32,) → 32 labels

class_names = ["glioma", "meningioma", "notumor", "pituitary"]

# Visualize 8 images from the batch
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("Batch Loading — Sample Batch", fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    img = images_batch[i].numpy()
    label = labels_batch[i].item()
    ax.imshow(img)
    ax.set_title(class_names[label], fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig("outputs/batch_sample.png", dpi=150, bbox_inches='tight')
plt.show()
print("Batch sample saved as outputs/batch_sample.png")


# =============================================
# STEP 5: Simulate Training Loop (No model yet)
# =============================================

print("\n Simulating batch streaming through full dataset...")
print("(This is exactly how training will work later)\n")

total_images_seen = 0

for batch_idx, (images, labels) in enumerate(train_loader):
    # In real training, your model would process here
    # For now we just count
    total_images_seen += len(images)
    
    if batch_idx % 10 == 0:
        print(f"Batch {batch_idx+1}/{len(train_loader)} "
              f"| Images seen so far: {total_images_seen}")

print(f"\nTotal images streamed: {total_images_seen}")
print("Streaming complete! RAM never filled up.")