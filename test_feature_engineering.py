import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer

# Load data
loader = DataLoader("data/bank_transactions_data_2.csv")
df = loader.load_data()

# Preprocess
processor = DataProcessor(df)
cleaned = processor.clean_data()
encoded = processor.encode_categorical()
scaled = processor.scale_features()

# Feature Engineering
engineer = FeatureEngineer(scaled)
final_df = engineer.full_feature_engineering()

print(final_df.head())
print("FINAL SHAPE:", final_df.shape)