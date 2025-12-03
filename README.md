# StoryOracle: Narrative Quality Analyzer

**One-line:** Lightweight NLP pipeline to analyze narrative quality in short fiction.

**Author:** Emmanuella Adams

## Quick Results
- Dataset: sample_data/story_samples.csv (30–100 paragraphs)
- Baseline: TF-IDF + Logistic Regression
- Features: readability, sentence length, lexical diversity, polarity
- Optional: DistilBERT fine-tune
- Sample report: reports/report.txt

## How to Run
1. Open `notebooks/01_data_prep.ipynb` in Colab
2. Install requirements: `pip install -r requirements.txt`
3. Run notebooks sequentially: Data prep → Model training → Evaluation
