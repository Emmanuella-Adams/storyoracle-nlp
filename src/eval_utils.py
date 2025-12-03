import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------------------
# Confusion Matrix
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# ------------------------------
# Paragraph-level Report
# ------------------------------
def generate_narrative_report(df, sample_ids=None):
    """
    Generate report per paragraph with key features
    """
    report_lines = []
    subset = df if sample_ids is None else df[df['id'].isin(sample_ids)]
    
    for _, row in subset.iterrows():
        line = (
            f"Paragraph {row['id']}: readability Flesch {row['flesch']:.1f}, "
            f"FK grade {row['fk_grade']:.1f}, avg sentence length {row['avg_sent_len']:.1f}, "
            f"lexical diversity {row['lexical_div']:.2f}, polarity {row['polarity']:.2f}."
        )
        report_lines.append(line)
    return report_lines

def save_report(report_lines, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in report_lines:
            f.write(line + "\n")
    print(f"Report saved to {path}")
