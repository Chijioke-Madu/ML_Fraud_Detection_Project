
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataProcessor:
    """Cleans and prepares dataset for model training."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.scaler = StandardScaler()

    def clean_data(self):
        # Drop duplicates and handle missing values
        self.df.drop_duplicates(inplace=True)
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        print("🧹 Data cleaned successfully.")
        return self.df

    def encode_categorical(self):
        # Encode categorical columns
        cat_cols = self.df.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in cat_cols:
            self.df[col] = le.fit_transform(self.df[col])
        print("🔠 Categorical features encoded.")
        return self.df

    def scale_features(self):
        num_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        self.df[num_cols] = self.scaler.fit_transform(self.df[num_cols])
        print("📏 Numerical features scaled.")
        return self.df
