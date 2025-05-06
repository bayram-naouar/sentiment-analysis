"""
model_trainer.py

Handles model tuning using GridSearchCV, loading/saving best parameters,
and training final models using the best configuration.
"""

import os
import json
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_param_grid(filepath: str, model_name: str) -> dict:
    """
    Loads hyperparameter grid from JSON. If file doesn't exist, creates and saves a default one.

    Parameters:
        filepath (str): Path to the JSON file.
        model_name (str): Model key name ('logreg', 'svm', or 'rf').

    Returns:
        dict: The hyperparameter grid.
    """
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)

    print(f"=> [INFO] No custom param_grid for {model_name}. Using and saving default.")

    if model_name == 'logreg':
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga'],
            'penalty': ['l1', 'l2'],
            'max_iter': [1000]
        }
    elif model_name == 'svm':
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    elif model_name == 'rf':
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        }
    else:
        raise ValueError(f"[ERROR] Invalid model name: {model_name}")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(param_grid, f, indent=4)

    return param_grid


def hyperparameter_tuning(clf, model_name: str, X_tune, y_tune):
    """
    Performs GridSearchCV and saves best parameters to JSON.

    Parameters:
        clf: The classifier instance.
        model_name (str): Identifier for model type.
        X_tune: Features for tuning.
        y_tune: Target labels for tuning.
    """
    print(f"=> [INFO] Tuning hyperparameters for {model_name}...")

    param_path = os.path.join(BASE_DIR, 'param_grids', f"{model_name}_param_grid.json")
    param_grid = load_param_grid(param_path, model_name)

    grid_search = GridSearchCV(
        clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1_macro',
        verbose=1
    )

    grid_search.fit(X_tune, y_tune)

    print(f"=> [INFO] Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"=> [INFO] Best f1_macro score: {grid_search.best_score_:.4f}")

    # Save best parameters
    os.makedirs(os.path.join(BASE_DIR, 'best_params'), exist_ok=True)
    with open(os.path.join(BASE_DIR, 'best_params', f"{model_name}_best_params.json"), 'w') as f:
        json.dump(grid_search.best_params_, f, indent=4)


def ask_yes_no(prompt: str, default: str = None) -> bool:
    """
    Prompt the user for a yes/no answer safely.

    Parameters:
        prompt (str): The message displayed to the user.
        default (str, optional): Default choice if user hits Enter ('y' or 'n').

    Returns:
        bool: True if user answers yes, False otherwise.
    """
    valid = {"yes": True, "y": True, "no": False, "n": False}

    if default:
        default = default.lower()
        if default not in valid:
            raise ValueError("Default must be 'y' or 'n'")
        prompt_suffix = " [Y/n] " if default == 'y' else " [y/N] "
    else:
        prompt_suffix = " [y/n] "

    while True:
        choice = input(prompt + prompt_suffix).strip().lower()
        if not choice and default:
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'y' or 'n' (or 'yes' or 'no').")


def tune_models(X_tune, y_tune):
    """
    Tune models on tuning data, optionally loading previously saved tuned models.

    Parameters:
        X_tune: Features for tuning.
        y_tune: Labels for tuning.
    """
    for model_name in ['logreg', 'svm', 'rf']:
        res = False
        best_params_path = os.path.join(BASE_DIR, 'best_params', f'{model_name}_best_params.json')

        if os.path.exists(best_params_path):
            res = ask_yes_no(f"=> [INFO] Found saved best_params for {model_name}. Use it? ", default='y')
        if res:
            continue
        else:
            if model_name == 'rf':
                _, X_tune_rf, _, y_tune_rf = train_test_split(X_tune, y_tune, stratify=y_tune, test_size=0.1, random_state=42)
                clf = RandomForestClassifier(random_state=42)
                hyperparameter_tuning(clf, model_name, X_tune_rf, y_tune_rf)
            elif model_name == 'svm':
                _, X_tune_svm, _, y_tune_svm = train_test_split(X_tune, y_tune, stratify=y_tune, test_size=0.1, random_state=42)
                clf = SVC(random_state=42)
                hyperparameter_tuning(clf, model_name, X_tune_svm, y_tune_svm)
            elif model_name == 'logreg':
                clf = LogisticRegression(random_state=42)
                hyperparameter_tuning(clf, model_name, X_tune, y_tune)
            else:
                raise ValueError(f"=> [ERROR] Invalid model name: {model_name}")
                


def fit_models_with_best_parameters(X_train, y_train):
    """
    Train final models using the best parameters found during tuning.

    Saves trained models into the 'models/' folder.

    Parameters:
        X_train: Training features.
        y_train: Training labels.
    """

    os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

    for model_name in ['logreg', 'svm', 'rf']:
        json_filepath = os.path.join(BASE_DIR, 'best_params', f"{model_name}_best_params.json")
        joblib_filepath = os.path.join(BASE_DIR, 'models', f"{model_name}_model_tuned.joblib")

        if os.path.exists(json_filepath):
            with open(json_filepath, 'r') as f:
                best_params = json.load(f)
            print(f"=> [INFO] Fitting {model_name}...")
            if model_name == 'logreg':
                clf = LogisticRegression(**best_params, random_state=42)
            elif model_name == 'svm':
                clf = SVC(**best_params, random_state=42)
            elif model_name == 'rf':
                clf = RandomForestClassifier(**best_params, random_state=42)
            else:
                raise ValueError(f"=> [ERROR] Invalid model name: {model_name}")
            clf.fit(X_train, y_train)
            joblib.dump(clf, joblib_filepath)
            print(f"=> [INFO] Saved: {joblib_filepath}")
        else:
            print(f"=> [ERROR] {json_filepath} not found")
