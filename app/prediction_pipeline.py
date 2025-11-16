import pandas as pd
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer

class PredictionPipeline:
    """Handles preprocessing + feature engineering for a single new transaction."""

    def __init__(self, model):
        self.model = model

    def preprocess(self, input_data: dict):
        # Convert input JSON to DataFrame
        df = pd.DataFrame([input_data])

        # Preprocessing
        processor = DataProcessor(df)
        cleaned = processor.clean_data()
        encoded = processor.encode_categorical()
        scaled = processor.scale_features()

        # Feature Engineering
        engineer = FeatureEngineer(scaled)
        final_df = engineer.full_feature_engineering()

        return final_df

    def predict(self, input_data: dict):
        final_df = self.preprocess(input_data)

        # Remove the target column if present
        if "FraudReported" in final_df.columns:
            final_df = final_df.drop(columns=["FraudReported"])

        prediction = self.model.predict(final_df)[0]
        return int(prediction)
