import pickle
import numpy as np
import os

models_dir = "models_exported"
stacker_path = os.path.join(models_dir, "stacker.pkl")

if os.path.exists(stacker_path):
    with open(stacker_path, 'rb') as f:
        stacker = pickle.load(f)
    print(f"Stacker coef shape: {stacker.coef_.shape}")
    print(f"Stacker coef: {stacker.coef_}")
else:
    print("Stacker not found")
