"""
preprocessor.py

Handles loading, cleaning, vectorizing, and splitting
the Sentiment140 dataset for sentiment analysis.
"""

import os
import re
import csv
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only once)
nltk.download('stopwords')
nltk.download('wordnet')

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_sentiment140_dataset(filepath: str, sample_size: float = 1.0) -> pd.DataFrame:
    """
    Load the Sentiment140 dataset and assign column names.

    Parameters:
        filepath (str): Path to dataset CSV.
        sample_size (float): Fraction of dataset to use (1.0 = full).

    Returns:
        pd.DataFrame: DataFrame with 'text' and mapped binary 'target'.
    """
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(filepath, encoding='latin-1', header=None, names=column_names)

    if sample_size < 1.0:
        df = df.sample(frac=sample_size, random_state=42)

    df['target'] = df['target'].astype(int).map({0: 0, 4: 1})

    return df[['text', 'target']]


def clean_tweet(text: str, lemmatizer: WordNetLemmatizer, stop_words: set) -> str:
    """
    Clean a tweet by removing URLs, mentions, digits, punctuation, and stopwords.

    Parameters:
        text (str): Raw tweet.
        lemmatizer (WordNetLemmatizer): Word lemmatizer.
        stop_words (set): Stopwords set.

    Returns:
        str: Cleaned tweet.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)     # Remove URLs
    text = re.sub(r"@\w+|#\w+", "", text)          # Remove mentions and hashtags
    text = re.sub(r"[^a-z\s]", "", text)           # Remove punctuation and digits

    words = text.split()
    cleaned = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(cleaned)


def vectorize_text(df: pd.DataFrame, max_features: int = 5000) -> tuple:
    """
    Clean and vectorize tweet text using TF-IDF.

    Parameters:
        df (pd.DataFrame): Dataset with raw 'text' and 'target' columns.
        max_features (int): TF-IDF vocabulary size.

    Returns:
        tuple: (X, y) where X is TF-IDF matrix, y is labels.
    """
    cleaned_data_path = os.path.join(BASE_DIR, 'data', 'cleaned_data.csv')

    if os.path.exists(cleaned_data_path):
        print(f"=> [INFO] Loading cleaned data from: {cleaned_data_path}")
        df = pd.read_csv(cleaned_data_path, encoding='latin-1', header=None, names=['clean_text', 'target'])
        df['target'] = df['target'].astype(int)
    else:
        print("=> [INFO] Cleaning raw tweets...")
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        df['clean_text'] = df['text'].apply(lambda x: clean_tweet(x, lemmatizer, stop_words))
        df = df[['clean_text', 'target']]
        df.to_csv(
            cleaned_data_path,
            encoding='latin-1',
            index=False,
            header=False,
            quotechar='"',
            quoting=csv.QUOTE_ALL
        )
        print(f"=> [INFO] Cleaned data saved to: {cleaned_data_path}")

    df = df.dropna(subset=['clean_text'])

    print("=> [INFO] Vectorizing text...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(df['clean_text'])

    y = df['target'].astype(int)

    return X, y
