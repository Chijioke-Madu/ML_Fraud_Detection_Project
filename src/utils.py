
# src/utils.py

import joblib
import os

def save_model(model, filename: str):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{filename}")
    print(f"✅ Model saved as models/{filename}")

def load_model(filename: str):
    return joblib.load(f"models/{filename}")
