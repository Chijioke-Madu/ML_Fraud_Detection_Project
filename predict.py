from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# Load model
MODEL_PATH = "models/best_model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return {
        "fraud_prediction": int(prediction),
        "fraud_probability": float(probability)
    }
