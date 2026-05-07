
import os
base_dir = os.path.dirname(__file__)
print(f"The base directory is:- {base_dir}")

train_path = os.path.join(base_dir, "..", "Datasets", "archive (7)", "Training")

# Coverting to absolute path
train_path = os.path.abspath(train_path)
print(f"The training path is:- {train_path}")

for categories in os.listdir(train_path):
    print(categories)

    
    

