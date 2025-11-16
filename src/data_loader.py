import pandas as pd

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self):
        """Loads a CSV file into a pandas DataFrame"""
        try:
            df = pd.read_csv(self.filepath)
            print(f"✅ Data loaded successfully with shape {df.shape}")
            return df
        except FileNotFoundError:
            print(f"❌ File not found at: {self.filepath}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
