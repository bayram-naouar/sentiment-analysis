"""
main.py

Entry point for the Sentiment Analysis project.
This script handles the full pipeline: loading data, preprocessing,
hyperparameter tuning, model training, and final evaluation.
"""

from src.preprocessor import load_sentiment140_dataset, vectorize_text
from src.model_trainer import tune_models, fit_models_with_best_parameters
from src.evaluator import evaluate_model
from src.distil_bert_trainer import train_distilbert_model, evaluate_distilbert_model

from sklearn.model_selection import train_test_split
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    """
    Main workflow for sentiment analysis pipeline.
    
    1. Loads the dataset.
    2. Cleans and vectorizes the text.
    3. Splits the data into training and test sets.
    4. Tunes hyperparameters for models.
    5. Trains the final models with the best parameters.
    6. Evaluates models on the test set.
    """
    '''
    # Step 1: Load dataset (sample_size=1.0 for full dataset)
    sample_size = 0.03
    print("[INFO] Loading dataset...")
    df = load_sentiment140_dataset('data/training.1600000.processed.noemoticon.csv', sample_size=sample_size)

    # Step 2: Clean and vectorize text, then split into features and labels
    print("[INFO] Cleaning, vectorizing text and splitting data...")
    X, y = vectorize_text(df)

    # Step 3: Split data into training and test sets
    print("[INFO] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    if sample_size == 1.0:
    # Take a small sample for hyperparameter tuning
        _, X_tune, _, y_tune = train_test_split(X, y, stratify=y, test_size=0.03, random_state=42)
    else:
        X_tune = X_train
        y_tune = y_train

    # Step 4: Tune hyperparameters for each model and save the best ones
    print("[INFO] Tuning hyperparameters...")
    tune_models(X_tune, y_tune)

    # Step 5: Train final models with the best hyperparameters using full training data
    print("[INFO] Training final models...")
    fit_models_with_best_parameters(X_train, y_train)

    # Step 6: Evaluate models on the test set
    print("[INFO] Evaluating models...")
    evaluate_model('models/logreg_model_tuned.joblib', X_test, y_test)
    evaluate_model('models/svm_model_tuned.joblib', X_test, y_test)
    evaluate_model('models/rf_model_tuned.joblib', X_test, y_test)
    '''

    # Using DistilBERT model (sample_size_bert=1.0 for full dataset)
    print("[INFO] Training / Loading DistilBERT model...")
    tokenized_dataset_test = train_distilbert_model(sample_size_bert=0.001)

    print("[INFO] Evaluating DistilBERT model...")
    evaluate_distilbert_model(tokenized_dataset_test['test']['text'], tokenized_dataset_test['test']['label'])

if __name__ == "__main__":
    main()
