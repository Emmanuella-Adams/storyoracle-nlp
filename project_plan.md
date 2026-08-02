# **StoryOracle Project Plan: Narrative Quality Analysis Pipeline**

## **1. Research Objective & Scope**
Emerging African fiction writers often face a lack of objective, automated diagnostic tools for evaluating paragraph-level narrative dynamics. StoryOracle addresses this gap by combining statistical text features (readability metrics, sentence structure, lexical richness, polarity) with TF-IDF vectorization and machine learning classifiers to evaluate tone, pacing, and structural flow in short fiction.

---

## **2. System Architecture & Workflow**

```
+--------------------------+
|  Short Fiction Paragraphs| (sample_data/story_samples.csv)
+--------------------------+
             |
             v
+--------------------------+
| Data Prep & Cleaning     | (src/data_utils.py)
| - Lowercasing & Sanitizing
| - NLTK Tokenization
+--------------------------+
             |
             v
+--------------------------+
| Feature Engineering      | (Flesch Ease, FK Grade, Avg Sent Length, TTR, Polarity)
+--------------------------+
             |
             v
+--------------------------+
| TF-IDF & Modeling        | (src/model_utils.py: Baseline Logistic Regression & RF)
+--------------------------+
             |
             v
+--------------------------+
| Evaluation & Reporting   | (src/eval_utils.py: Figures + Diagnostic Reports)
+--------------------------+
```

---

## **3. Key Research Deliverables**
1. **Reproducible Python Library:** Modular utilities (`data_utils.py`, `model_utils.py`, `eval_utils.py`).
2. **Interactive Jupyter Notebooks:** `01_data_prep.ipynb`, `02_model_training.ipynb`, `03_evaluation_visuals.ipynb`.
3. **Narrative Diagnostic Report:** Paragraph-level actionable feedback (`reports/report.txt`).
4. **Visual Analytics:** High-resolution distribution plots and performance metrics (`figures/`).

---

## **4. Future Directions**
* Integration of transformer fine-tuning (DistilBERT / RoBERTa) for fine-grained emotion multi-label classification.
* Support for multilingual African literature text samples.
