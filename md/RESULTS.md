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