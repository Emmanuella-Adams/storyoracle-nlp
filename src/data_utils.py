import re
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are available with fallback
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

def safe_sent_tokenize(text):
    try:
        return sent_tokenize(text)
    except Exception:
        # Fallback simple sentence splitting
        sents = re.split(r'[.!?]+', text)
        return [s.strip() for s in sents if s.strip()]

def safe_word_tokenize(text):
    try:
        return word_tokenize(text)
    except Exception:
        # Fallback simple word tokenization
        return re.findall(r'\b\w+\b', text.lower())

# ------------------------------
# Data Cleaning
# ------------------------------
def clean_text(text):
    """
    Clean raw text: lowercase, remove URLs, remove extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clean_csv(csv_path):
    """
    Load CSV dataset and standardize column names.
    Supports both 'text' and 'Sentence' column names.
    """
    df = pd.read_csv(csv_path)
    
    # Standardize text column
    if 'text' in df.columns:
        text_col = 'text'
    elif 'Sentence' in df.columns:
        text_col = 'Sentence'
        df['text'] = df['Sentence']
    else:
        text_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        df['text'] = df[text_col]

    # Standardize ID column
    if 'id' not in df.columns:
        if 'Index' in df.columns:
            df['id'] = df['Index']
        else:
            df['id'] = range(1, len(df) + 1)

    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].apply(lambda x: len(safe_word_tokenize(x))) > 3].reset_index(drop=True)

    # Derive primary emotion label if multi-emotion probability columns exist
    emotion_cols = [
        'Caring', 'Boredom', 'Disappointment', 'Nervousness', 'Annoyance', 'Contempt',
        'Sarcasm', 'Fear', 'Approval', 'Desire', 'Curiosity', 'Disgust', 'Pride',
        'Confusion', 'Gratitude', 'Love', 'Amusement', 'Grief', 'Joy', 'Admiration',
        'Embarrassment', 'Neutral', 'Anger', 'Disapproval', 'Relief', 'Surprise',
        'Remorse', 'Realization', 'Excitement', 'Envy', 'Optimism', 'Sadness'
    ]
    present_emotions = [col for col in emotion_cols if col in df.columns]
    if present_emotions and 'label' not in df.columns:
        df['label'] = df[present_emotions].idxmax(axis=1)

    return df

# ------------------------------
# Paragraph Features
# ------------------------------
def paragraph_features(text):
    """
    Extract numeric narrative quality and linguistic features from a paragraph.
    """
    sents = safe_sent_tokenize(text)
    words = safe_word_tokenize(text)
    num_words = len(words)
    num_sents = max(len(sents), 1)

    avg_sent_len = num_words / num_sents
    lexical_div = len(set(words)) / num_words if num_words > 0 else 0.0

    try:
        flesch = textstat.flesch_reading_ease(text)
        fk_grade = textstat.flesch_kincaid_grade(text)
    except Exception:
        flesch = 60.0
        fk_grade = 8.0

    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity = 0.0
        subjectivity = 0.0

    return {
        'avg_sent_len': avg_sent_len,
        'lexical_div': lexical_div,
        'num_sents': num_sents,
        'flesch': flesch,
        'fk_grade': fk_grade,
        'polarity': polarity,
        'subjectivity': subjectivity
    }

def add_features(df):
    """
    Apply feature extraction across dataframe paragraphs.
    """
    feat_df = df['clean_text'].apply(paragraph_features).apply(pd.Series)
    for col in feat_df.columns:
        if col in df.columns:
            df[col] = feat_df[col]
        else:
            df = pd.concat([df, feat_df[[col]]], axis=1)
    return df

# ------------------------------
# TF-IDF Vectorization
# ------------------------------
def tfidf_vectorize(df, max_features=1000, vectorizer=None):
    """
    Convert clean text to TF-IDF feature matrix.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
        X_tfidf = vectorizer.fit_transform(df['clean_text'])
    else:
        X_tfidf = vectorizer.transform(df['clean_text'])
    return X_tfidf, vectorizer
