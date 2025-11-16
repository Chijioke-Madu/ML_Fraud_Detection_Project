import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class ModelTrainer:
    """Trains multiple ML models and selects the best performer."""

    def __init__(self, df: pd.DataFrame, target_column: str):
        self.df = df.copy()
        self.target = target_column
        self.X = self.df.drop(columns=[self.target])
        self.y = self.df[self.target]
        self.models = {}
        self.results = {}

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print("🔀 Data split into train and test sets.")

    def train_logistic_regression(self):
        model = LogisticRegression(max_iter=200)
        model.fit(self.X_train, self.y_train)
        self.models["Logistic Regression"] = model
        print("📘 Logistic Regression trained.")

    def train_random_forest(self):
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models["Random Forest"] = model
        print("🌲 Random Forest trained.")

    def evaluate_models(self):
        for name, model in self.models.items():
            preds = model.predict(self.X_test)
            metrics = {
                "accuracy": accuracy_score(self.y_test, preds),
                "precision": precision_score(self.y_test, preds),
                "recall": recall_score(self.y_test, preds),
                "f1-score": f1_score(self.y_test, preds),
            }
            self.results[name] = metrics
            print(f"📊 {name} evaluation:", metrics)

    def select_best_model(self):
        best_model_name = max(self.results, key=lambda m: self.results[m]["f1-score"])
        best_model = self.models[best_model_name]
        print(f"🏆 Best model selected: {best_model_name}")
        return best_model

    def save_model(self, model, filename="best_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 Model saved as {filename}")

    def run_all(self):
        self.split_data()
        self.train_logistic_regression()
        self.train_random_forest()
        self.evaluate_models()
        best_model = self.select_best_model()
        self.save_model(best_model)
        return best_model, self.results
