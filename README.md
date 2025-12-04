
# **StoryOracle: Narrative Quality Analyzer for Short Fiction**

**One-line:** A lightweight NLP pipeline to analyze narrative quality in short fiction, providing interpretable feedback on readability, tone, pacing, and style.

**Author:** Emmanuella Adams

---

## **Project Overview**

StoryOracle is designed to support emerging African writers by automatically analyzing short fiction paragraphs. The system evaluates **readability**, **emotional tone**, **sentence structure**, and **lexical diversity**, producing a structured report per paragraph. It uses a **TF-IDF + Logistic Regression baseline**, with optional transformer fine-tuning (DistilBERT/BERT) for nuanced emotion classification.

Key goals:

* Provide **actionable narrative feedback** for writers.
* Demonstrate **reproducible student-level NLP research**.
* Align with **human-centered AI priorities** for African content creators.

---

## **Repository Structure**

```
/story-sensei-nlp
├─ README.md
├─ requirements.txt
├─ notebooks/
│  ├─ 01_data_prep.ipynb         # Clean, tokenize, extract features
│  ├─ 02_model_training.ipynb    # Train TF-IDF + Logistic Regression baseline
│  ├─ 03_evaluation_visuals.ipynb # Evaluate, visualize, and generate report
├─ src/
│  ├─ data_utils.py              # Text cleaning & feature extraction
│  ├─ model_utils.py             # Baseline model, save/load functions
│  ├─ eval_utils.py              # Confusion matrix & paragraph-level reports
├─ reports/
│  └─ report.txt                 # Example narrative report
├─ figures/
│  └─ ...png                     # Histograms, confusion matrix, etc.
├─ sample_data/
│  └─ story_samples.csv          # 30–100 paragraphs of text
└─ project_plan.md
```

---

## **Dataset**

* File: `sample_data/story_samples.csv`
* Columns: `id`, `text`, `label` etc.
* 30–100 paragraphs exported from a short fiction made by me.
* Cleaned, tokenized, and feature-enhanced in `01_data_prep.ipynb`.

---

## **Features Extracted**

For each paragraph:

* **Readability:** Flesch Reading Ease, Flesch-Kincaid grade
* **Sentence metrics:** average sentence length, number of sentences
* **Lexical diversity:** unique word ratio
* **Polarity / Sentiment:** via TextBlob
* **Optional:** Emotion labels

---

## **Baseline Modeling**

* **TF-IDF + Logistic Regression** on paragraph text.
* Optional transformer fine-tune: DistilBERT/BERT for emotion classification.
* Train-test split: 80–20, reproducible with fixed random seed.

Evaluation metrics:

* Accuracy
* F1-score
* Confusion matrix
* Qualitative report: narrative-level feedback

---

## **Usage**

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Data Preprocessing

Open `notebooks/01_data_prep.ipynb` and run:

* Load CSV
* Clean text
* Extract features

### 3️⃣ Model Training

Open `notebooks/02_model_training.ipynb` and run:

* TF-IDF vectorization
* Train baseline model
* Save trained model (`model_utils.save_model`)

### 4️⃣ Evaluation & Report Generation

Open `notebooks/03_evaluation_visuals.ipynb` and run:

* Evaluate predictions
* Plot confusion matrix
* Generate paragraph-level narrative report (`eval_utils.generate_narrative_report`)
* Save figures to `figures/`

---

## **Example Narrative Report**

```
Paragraph 1: readability Flesch 72.4, FK grade 6.2, avg sentence length 12.4, lexical diversity 0.65, polarity 0.12.
Paragraph 2: readability Flesch 58.1, FK grade 8.0, avg sentence length 18.2, lexical diversity 0.55, polarity -0.05.
...
```

---

## **Figures Saved**

* Histogram of Flesch Reading Ease
* Average sentence length distribution
* Lexical diversity distribution
* Confusion matrix
* Sample narrative paragraph reports

---

## **Contributing**

1. Fork the repo
2. Create a new branch (`feature/your-feature`)
3. Add features or improve models
4. Submit a pull request with clear description

---

## **Future Improvements**

* Expand dataset for more paragraphs and labels
* Add **African-language support**
* Incorporate **human-in-the-loop feedback**
* Integrate transformer-based emotion classification

---

## **References & Tools**

* [NLTK](https://www.nltk.org/) — Tokenization & text processing
* [TextBlob](https://textblob.readthedocs.io/) — Sentiment analysis
* [textstat](https://pypi.org/project/textstat/) — Readability metrics
* [Scikit-learn](https://scikit-learn.org/) — TF-IDF & Logistic Regression
* A short fiction, "The Price Of Dusk", written by Emmanuella Adams to extract the data samples.


