import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import textstat
from textblob import TextBlob
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')

# ------------------------------
# Data Cleaning
# ------------------------------
def clean_text(text):
    """
    Clean raw text: lowercase, remove URLs, remove extra whitespace
    """
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_clean_csv(csv_path):
    """
    Load CSV and clean text
    """
    df = pd.read_csv(csv_path)
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.split().apply(len) > 3]  # remove tiny paragraphs
    return df

# ------------------------------
# Paragraph Features
# ------------------------------
def paragraph_features(text):
    """
    Extract numeric features from paragraph for analysis
    """
    sents = sent_tokenize(text)
    words = word_tokenize(text)
    avg_sent_len = sum(len(s.split()) for s in sents) / len(sents)
    lexical_div = len(set(words)) / len(words) if len(words) > 0 else 0
    num_sents = len(sents)
    flesch = textstat.flesch_reading_ease(text)
    fk_grade = textstat.flesch_kincaid_grade(text)
    polarity = TextBlob(text).sentiment.polarity
    return {
        'avg_sent_len': avg_sent_len,
        'lexical_div': lexical_div,
        'num_sents': num_sents,
        'flesch': flesch,
        'fk_grade': fk_grade,
        'polarity': polarity
    }

def add_features(df):
    """
    Apply paragraph_features to DataFrame
    """
    feat_df = df['clean_text'].apply(paragraph_features).apply(pd.Series)
    df = pd.concat([df, feat_df], axis=1)
    return df

# ------------------------------
# TF-IDF Vectorization
# ------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(df, max_features=1000):
    """
    Convert text to TF-IDF features
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    return X_tfidf, vectorizer
