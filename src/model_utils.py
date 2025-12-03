from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# ------------------------------
# Baseline Model (TF-IDF + LR)
# ------------------------------
def train_baseline(X, y, test_size=0.2, random_state=42):
    """
    Train Logistic Regression baseline on TF-IDF or feature matrix
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    return clf, report, X_train, X_test, y_train, y_test

# ------------------------------
# Save & Load Models
# ------------------------------
def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model