import pandas as pd
import numpy as np

class FeatureEngineer:
    """Creates new predictive features for fraud detection."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def create_time_features(self):
        """Extracts time-related features from TransactionDate."""
        if "TransactionDate" in self.df.columns:
            self.df["TransactionHour"] = pd.to_datetime(self.df["TransactionDate"]).dt.hour
            self.df["TransactionDay"] = pd.to_datetime(self.df["TransactionDate"]).dt.dayofweek
            print("⏰ Time features extracted.")
        return self.df

    def create_amount_features(self):
        """Creates features based on transaction amounts."""
        if "TransactionAmount" in self.df.columns:
            # log1p() handles zeros but NOT negative values — we clean those
            self.df["TransactionAmountPos"] = self.df["TransactionAmount"].apply(lambda x: max(x, 0))
            self.df["LogTransactionAmount"] = np.log1p(self.df["TransactionAmountPos"])
            
            median_val = self.df["TransactionAmount"].median()
            self.df["IsLargeTransaction"] = (self.df["TransactionAmount"] > median_val).astype(int)
            
            print("💰 Amount-based features created.")
        return self.df

    def create_behavioral_features(self):
        """Generates behavioral features for each user."""
        if all(col in self.df.columns for col in ["AccountID", "TransactionAmount"]):
            self.df["AvgUserTransaction"] = self.df.groupby("AccountID")["TransactionAmount"].transform("mean")
            self.df["TransactionDeviation"] = self.df["TransactionAmount"] - self.df["AvgUserTransaction"]
            print("🧠 Behavioral features created.")
        return self.df

    def create_target_label(self):
        """Generate a synthetic fraud label based on realistic fraud patterns."""
        self.df["FraudReported"] = 0

        # Rule-based synthetic fraud detection
        # (These patterns mimic real-world fraud flags)
        self.df.loc[self.df["IsLargeTransaction"] == 1, "FraudReported"] = 1
        self.df.loc[self.df["TransactionDeviation"] > 2, "FraudReported"] = 1
        self.df.loc[self.df["TransactionHour"].isin([0, 1, 2, 3]), "FraudReported"] = 1
        self.df.loc[self.df["LoginAttempts"] > 3, "FraudReported"] = 1

        print("🚨 Synthetic fraud label created.")
        return self.df

    def full_feature_engineering(self):
        """Runs all feature engineering steps including synthetic target creation."""
        self.create_time_features()
        self.create_amount_features()
        self.create_behavioral_features()
        self.create_target_label()
        print("✨ All feature engineering steps completed.")
        return self.df
