import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------------------
# Visualization Utilities
# ------------------------------
def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plot and optionally save styled confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('Actual Label', fontsize=12)
    plt.title('Emotion / Narrative Quality Confusion Matrix', fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.close()

def plot_feature_distributions(df, save_dir=None):
    """
    Plot histograms and KDE curves for key text metrics.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    
    os.makedirs(os.path.abspath(save_dir), exist_ok=True)

    # 1. Flesch Reading Ease Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(df['flesch'], kde=True, color='teal', bins=15)
    plt.title('Flesch Reading Ease Distribution', fontsize=13)
    plt.xlabel('Flesch Reading Ease Score', fontsize=11)
    plt.ylabel('Paragraph Count', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'flesch_reading_ease_dist.png'), dpi=300)
    plt.close()

    # 2. Average Sentence Length Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(df['avg_sent_len'], kde=True, color='coral', bins=15)
    plt.title('Average Sentence Length Distribution', fontsize=13)
    plt.xlabel('Words per Sentence', fontsize=11)
    plt.ylabel('Paragraph Count', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sentence_length_dist.png'), dpi=300)
    plt.close()

    # 3. Lexical Diversity Distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(df['lexical_div'], kde=True, color='purple', bins=15)
    plt.title('Lexical Diversity (Type-Token Ratio) Distribution', fontsize=13)
    plt.xlabel('Type-Token Ratio', fontsize=11)
    plt.ylabel('Paragraph Count', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'lexical_diversity_dist.png'), dpi=300)
    plt.close()
    print(f"Feature distribution plots saved to {save_dir}")

# ------------------------------
# Paragraph-level Narrative Report
# ------------------------------
def generate_narrative_report(df, sample_ids=None):
    """
    Generate actionable per-paragraph narrative diagnostics report.
    """
    report_lines = [
        "==========================================================================",
        "          STORYORACLE: NARRATIVE QUALITY DIAGNOSTIC REPORT",
        "==========================================================================\n"
    ]
    subset = df if sample_ids is None else df[df['id'].isin(sample_ids)]
    
    for _, row in subset.iterrows():
        paragraph_id = row['id']
        flesch = row.get('flesch', 0.0)
        fk_grade = row.get('fk_grade', 0.0)
        avg_len = row.get('avg_sent_len', 0.0)
        lex_div = row.get('lexical_div', 0.0)
        polarity = row.get('polarity', 0.0)
        label = row.get('label', 'N/A')

        pacing_feedback = "Fast-paced & concise" if avg_len < 14 else ("Balanced" if avg_len <= 22 else "Dense & descriptive")
        tone_feedback = "Positive" if polarity > 0.1 else ("Negative" if polarity < -0.1 else "Neutral")

        line = (
            f"Paragraph {paragraph_id} [Label: {label}]:\n"
            f"  - Readability: Flesch Ease = {flesch:.1f}, FK Grade Level = {fk_grade:.1f}\n"
            f"  - Sentence Structure: Avg Sentence Length = {avg_len:.1f} words ({pacing_feedback})\n"
            f"  - Lexical Diversity: Unique Word Ratio = {lex_div:.2f}\n"
            f"  - Emotional Polarity: {polarity:.2f} ({tone_feedback} Tone)\n"
        )
        report_lines.append(line)
        
    return report_lines

def save_report(report_lines, path=None):
    """
    Save report text lines to file.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'report.txt')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for line in report_lines:
            f.write(line + "\n")
    print(f"Narrative diagnostic report successfully saved to {path}")
