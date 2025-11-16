import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer

# Load data
loader = DataLoader("data/bank_transactions_data_2.csv")
df = loader.load_data()

# Preprocessing
processor = DataProcessor(df)
cleaned = processor.clean_data()
encoded = processor.encode_categorical()
scaled = processor.scale_features()

# Feature engineering
engineer = FeatureEngineer(scaled)
final_df = engineer.full_feature_engineering()

# Model Training
trainer = ModelTrainer(final_df, target_column="FraudReported")
best_model, results = trainer.run_all()

print("\n🏆 FINAL RESULTS:")
print(results)
