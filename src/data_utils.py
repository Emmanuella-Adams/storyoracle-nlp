import re
import pandas as pd
from nltk.tokenize import sent_tokenize

def split_into_paragraphs(text, min_sent=3, max_sent=7):
    sentences = sent_tokenize(text)
    paragraphs = []
    current = []

    for sent in sentences:
        current.append(sent)

        # When we hit 3–7 sentences, start a new paragraph
        if len(current) >= min_sent:
            if len(current) <= max_sent:
                paragraphs.append(" ".join(current))
                current = []

    # leftover sentences
    if current:
        paragraphs.append(" ".join(current))

    return paragraphs

def build_csv(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = split_into_paragraphs(text)
    df = pd.DataFrame({
        "id": range(1, len(paragraphs)+1),
        "text": paragraphs,
        "label": [""] * len(paragraphs)  # optional
    })

    df.to_csv(output_path, index=False)
    print(f"Generated {len(paragraphs)} paragraphs -> {output_path}")

