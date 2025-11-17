import os
import pandas as pd
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer

DATA_PATH = "data/bank_transactions_data_2.csv"
MODEL_PATH = "models/best_model.pkl"

def main():
    print("🚀 Starting training pipeline...")

    loader = DataLoader(DATA_PATH)
    df = loader.load_data()

    processor = DataProcessor(df)
    df = processor.clean_data()
    df = processor.encode_categorical()
    df = processor.scale_features()

    fe = FeatureEngineer(df)
    df = fe.run_all()

    trainer = ModelTrainer(df, target_column="FraudReported")
    results = trainer.train_all_models(save_path=MODEL_PATH)

    print("🎉 Training completed!")
    print(results)


if __name__ == "__main__":
    main()
