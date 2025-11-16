import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import DataLoader
from src.data_processor import DataProcessor

loader = DataLoader("data/bank_transactions_data_2.csv")
df = loader.load_data()

processor = DataProcessor(df)
cleaned_df = processor.clean_data()
encoded_df = processor.encode_categorical()
scaled_df = processor.scale_features()

print(scaled_df.head())
