from fastapi import FastAPI
import pickle
from app.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Fraud Detection API")

# Load model at startup
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

pipeline = PredictionPipeline(model)

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict_transaction(data: dict):
    """Accepts JSON input and returns fraud prediction."""
    try:
        prediction = pipeline.predict(data)
        return {
            "fraud_prediction": prediction,
            "message": "1 means FRAUD, 0 means NOT FRAUD"
        }
    except Exception as e:
        return {"error": str(e)}
