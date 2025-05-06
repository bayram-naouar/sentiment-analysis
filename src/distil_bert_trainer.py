"""
distilbert_trainer.py

Fine-tunes DistilBERT on sentiment classification using Hugging Face Transformers.
"""

import os

import torch
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from src.preprocessor import load_sentiment140_dataset

# Ensure reproducibility
torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def prepare_data(df, tokenizer, max_length=128):
    """
    Tokenizes the dataset for use with DistilBERT.

    Args:
        df (pd.DataFrame): DataFrame with 'text' and 'target' columns.
        tokenizer: HuggingFace tokenizer.
        max_length (int): Max token length for padding/truncation.

    Returns:
        Dataset: Tokenized Hugging Face Dataset.
    """
    # rename to Hugging Face convention
    df['label'] = df['target'] 
    hf_dataset = Dataset.from_pandas(df[['text', 'label']])

    def tokenize(batch):
        return tokenizer(batch['text'], padding="max_length", truncation=True, max_length=max_length)

    return hf_dataset.map(tokenize, batched=True)


def train_distilbert_model(sample_size_bert=0.001):
    """
    Loads data, tokenizes, fine-tunes DistilBERT and saves the model.
    """
    print("=> [INFO] Loading and sampling data...")
    filepath = os.path.join(BASE_DIR, 'data', 'training.1600000.processed.noemoticon.csv')
    df = load_sentiment140_dataset(filepath, sample_size=sample_size_bert)

    print("=> [INFO] Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    print("=> [INFO] Tokenizing dataset...")
    tokenized_dataset = prepare_data(df, tokenizer)

    print("=> [INFO] Splitting into train and validation sets...")
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

    print("=> [INFO] Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    model_path_dir = os.path.join(BASE_DIR, 'models', 'distilbert_sentiment')
    logs_path_dir = os.path.join(BASE_DIR, 'logs', 'distilbert_sentiment')
    training_args = TrainingArguments(
        output_dir=model_path_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=logs_path_dir,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        disable_tqdm=False,
        logging_first_step=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    print("=> [INFO] Starting training...")
    trainer.train()

    print("=> [INFO] Saving model and tokenizer...")
    model.save_pretrained(model_path_dir)
    tokenizer.save_pretrained(model_path_dir)
    print("=> [DONE] DistilBERT model fine-tuned and saved.")
    return tokenized_dataset["test"]


def evaluate_distilbert_model(texts, labels):
    """
    Loads a trained DistilBERT model and evaluates it on given test data.

    Args:
        texts (List[str]): Raw input texts.
        labels (List[int]): True sentiment labels.

    Returns:
        None
    """
    print("=> [INFO] Loading model and tokenizer for evaluation...")
    model_path_dir = os.path.join(BASE_DIR, 'models', 'distilbert_sentiment')
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path_dir)
    model = DistilBertForSequenceClassification.from_pretrained(model_path_dir)

    print("=> [INFO] Tokenizing test data...")
    encoded = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    model.eval()
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits
        predictions = torch.argmax(logits, axis=1).numpy()

    print("=> [INFO] Evaluation results:")
    print("Accuracy:", accuracy_score(labels, predictions))
    print("Classification Report:\n", classification_report(labels, predictions))

    cm = confusion_matrix(labels, predictions, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('DistilBERT Normalized Confusion Matrix')

    print("=> [INFO] Saving results:")
    plt.savefig(os.path.join(BASE_DIR, 'results', 'distilbert_sentiment', 'distilbert_confusion_matrix.png'))
    plt.show()