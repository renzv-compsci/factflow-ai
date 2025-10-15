# Fake News Classifier Evaluation Tables

This document summarizes the classifier evaluation tables generated throughout our development and testing process. It is intended for tracking model progress, comparison, and error analysis.

---

## 1. **Initial Perfect Scores (Trivial Dataset)**

| Metric      | Class 0 (Fake) | Class 1 (Real) | Overall |
|-------------|---------------|---------------|---------|
| Precision   | 1.00          | 1.00          | 1.00    |
| Recall      | 1.00          | 1.00          | 1.00    |
| F1-score    | 1.00          | 1.00          | 1.00    |
| Accuracy    |               |               | 1.00    |
| Support     | 23481         | 21417         | 44898   |

*Notes: Model trained on separate fake.csv and true.csv files. Results are artificially perfect due to trivial file-based separation.*

---

## 2. **First Realistic Test (Scrambled/Mixed Dataset)**

| Metric      | Class 0 (Fake) | Class 1 (Real) | Overall    |
|-------------|---------------|---------------|------------|
| Precision   | 0.62          | 0.63          | (avg: 0.62)|
| Recall      | 0.38          | 0.82          | (avg: 0.60)|
| F1-score    | 0.47          | 0.71          | (avg: 0.59)|
| Accuracy    |               |               | 0.63       |
| Support     | 4488          | 5752          | 10240      |

*Notes: Model trained and evaluated on properly mixed data. Shows realistic strengths/weaknesses, especially lower recall for fake news.*

---

## 3. **Improved Model – Bias Towards Fake News**

| Metric      | Class 0 (Fake) | Class 1 (Real) | Overall    |
|-------------|---------------|---------------|------------|
| Precision   | 0.44          | 0.63          | (avg: 0.53)|
| Recall      | 0.95          | 0.07          | (avg: 0.51)|
| F1-score    | 0.60          | 0.12          | (avg: 0.36)|
| Accuracy    |               |               | 0.45       |
| Support     | 4488          | 5752          | 10240      |

*Notes: Very high recall for fake news, very low for real news. Indicates strong bias toward predicting fake news.*

---

# Fake News Classifier Evaluation Report

**Date:** 2025-10-15  
**Model:** Logistic Regression Pipeline (latest)  
**Test Set:** `train_mapped.tsv`

---

## Classification Report

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Fake) | 0.44 | 0.93 | 0.59 | 4488 |
| 1 (Real) | 0.53 | 0.06 | 0.10 | 5752 |

**Overall Accuracy:** 0.44 (44%)  
**Macro Avg:** Precision 0.48, Recall 0.50, F1-score 0.35  
**Weighted Avg:** Precision 0.49, Recall 0.44, F1-score 0.32  
**Total support:** 10240

---

## Interpretation

- The model is heavily biased toward predicting "Fake" (class 0), capturing most fake examples (recall 0.93) but missing almost all "Real" news (recall 0.06).
- Overall accuracy is **44%**, which is below random guessing for a balanced dataset.
- The F1-score for "Real" news is particularly low (0.10), indicating a need for substantial improvement in correctly identifying true news articles.

---