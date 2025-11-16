
from sklearn.metrics import classification_report, confusion_matrix

class ModelEvaluator:
    """Evaluates trained models with standard metrics."""

    @staticmethod
    def evaluate(model, X_test, y_test):
        preds = model.predict(X_test)
        print("📊 Classification Report:")
        print(classification_report(y_test, preds))
        print("🧾 Confusion Matrix:")
        print(confusion_matrix(y_test, preds))
