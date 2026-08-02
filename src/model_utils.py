import os
import joblib
import pickle
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score

# ------------------------------
# Baseline & Narrative Classifiers
# ------------------------------
def train_baseline(X, y, model_type='logistic_regression', test_size=0.2, random_state=42):
    """
    Train a classification baseline on text/numerical feature matrix.
    Supports LogisticRegression and RandomForestClassifier.
    Automatically handles small class counts without failing stratify.
    """
    counts = Counter(y)
    min_count = min(counts.values()) if counts else 0
    use_stratify = y if min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=use_stratify
    )

    if model_type == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        clf = LogisticRegression(max_iter=1000, random_state=random_state)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    report = classification_report(y_test, y_pred, zero_division=0)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'report': report
    }

    return clf, metrics, X_train, X_test, y_train, y_test

def evaluate_cross_validation(clf, X, y, cv=5):
    """
    Perform Stratified K-Fold cross validation and return mean accuracy and std.
    """
    try:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        return np.mean(scores), np.std(scores)
    except Exception:
        return 0.0, 0.0

# ------------------------------
# Save & Load Models
# ------------------------------
def save_model(model, path):
    """
    Save trained model or pipeline artifact.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model successfully saved to {path}")

def load_model(path):
    """
    Load saved model artifact.
    """
    model = joblib.load(path)
    print(f"Model loaded from {path}")
    return model