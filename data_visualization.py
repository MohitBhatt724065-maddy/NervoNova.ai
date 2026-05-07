# Dataset visualization on processed data.
import numpy as np
import matplotlib.pyplot as plt

shape = np.load("Processed_Data/train_shape.npy")
train_X = np.memmap("Processed_Data/train_X.dat", dtype='float32', mode='r', shape=tuple(shape))
train_y = np.fromfile("Processed_Data/train_y.dat", dtype='int32')

class_name = ["glioma", "meningioma", "notumor", "pituitary"]

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('Brain Tumor MRI - Sample Dataset', fontsize=18, fontweight='bold')

for i, ax in enumerate(axes.flatten()):
    img = train_X[i]
    img = (img - img.min()) / (img.max() - img.min())
    ax.imshow(img)
    ax.set_title(class_name[train_y[i]], fontsize=10)
    ax.axis('off')
    
plt.tight_layout()
plt.savefig("brain_tumor_samples.png", dpi=150, bbox_inches='tight')
plt.show()
print("Sample images saved as brain_tumor_samples.png")