"""
evaluator.py

Evaluates trained models using accuracy, classification report, and a confusion matrix heatmap.
"""

import os
import joblib
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def evaluate_model(model_path, X_test, y_test):
    """
    Evaluate the trained model on the test set.

    Parameters:
        model_path (str): Path to the saved model file (.joblib).
        X_test (numpy.ndarray): Test features.
        y_test (numpy.ndarray): True labels for test set.

    Returns:
        None
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found at: {model_path}")

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    print("=> Accuracy on test set:", accuracy_score(y_test, y_pred))
    print("=> Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix (normalized per true class)
    cm = confusion_matrix(y_test, y_pred, normalize='true')

    labels = ['Negative', 'Positive']

    # Plot
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Normalized Confusion Matrix')
    plt.show()
