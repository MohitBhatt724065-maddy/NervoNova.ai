# What i exactlt did i save in Day 3?
import numpy as np
import os

train_shape = np.load("Processed_Data/train_shape.npy")
test_shape = np.load("Processed_Data/test_shape.npy")

print("Training Data!")
print(f"Total images : {train_shape[0]}")
print(f"Image Size : {train_shape[1]} x {train_shape[2]}")
print(f"Color Channels : {train_shape[3]}")

print("Testing Data!")
print(f"Total images : {test_shape[0]}")
print(f"Image Size : {test_shape[1]} x {test_shape[2]}")
print(f"Color Channels : {test_shape[[3]]}" )


for f in ["test_X.dat", "test_y.dat", "train_X.dat", "train_y.dat"]:
    path = f"Processed_Data/{f}"
    size = os.path.getsize(path)/(1024**3)
    print(f"{f} : {size:.2f} GB")