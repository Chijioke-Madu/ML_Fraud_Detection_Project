from src.data_loader import DataLoader

loader = DataLoader("data/bank_transactions_data_2.csv")
df = loader.load_data()

print("\nRAW COLUMNS:")
print(df.columns)
