# Fake News Classifier Evaluation Tables

This document summarizes the classifier evaluation tables generated throughout our development and testing process. It is intended for tracking model progress, comparison, and error analysis.

---

## 1. **Initial Perfect Scores (Trivial Dataset)**

| Metric      | Class 0 (Fake) | Class 1 (Real) | Overall |
|-------------|---------------:|---------------:|--------:|
| Precision   | 1.00           | 1.00           | 1.00    |
| Recall      | 1.00           | 1.00           | 1.00    |
| F1-score    | 1.00           | 1.00           | 1.00    |
| Accuracy    |                |                | 1.00    |
| Support     | 23481          | 21417          | 44898   |

*Notes: Model trained on separate fake.csv and true.csv files. Results are artificially perfect due to trivial file-based separation.*

---

## 2. **First Realistic Test (Scrambled/Mixed Dataset)**

| Metric      | Class 0 (Fake) | Class 1 (Real) | Overall    |
|-------------|---------------:|---------------:|-----------:|
| Precision   | 0.62           | 0.63           | 0.62       |
| Recall      | 0.38           | 0.82           | 0.60       |
| F1-score    | 0.47           | 0.71           | 0.59       |
| Accuracy    |                |                | 0.63       |
| Support     | 4488           | 5752           | 10240      |

*Notes: Model trained and evaluated on properly mixed data. Shows realistic strengths/weaknesses, especially lower recall for fake news.*

---

## 3. **Improved Model – Bias Towards Fake News**

| Metric      | Class 0 (Fake) | Class 1 (Real) | Overall    |
|-------------|---------------:|---------------:|-----------:|
| Precision   | 0.44           | 0.63           | 0.53       |
| Recall      | 0.95           | 0.07           | 0.51       |
| F1-score    | 0.60           | 0.12           | 0.36       |
| Accuracy    |                |                | 0.45       |
| Support     | 4488           | 5752           | 10240      |

*Notes: Very high recall for fake news, very low for real news. Indicates strong bias toward predicting fake news.*

---

# Fake News Classifier Evaluation Report

**Date:** 2025-11-23  
**Model:** Logistic Regression Pipeline (grid-searched TF-IDF + Logistic)  
**Test Set:** `train_mapped_clean.tsv` (canonical raw-input evaluation)

---

## 4. **Current Best Model (Post-fixes)**

After unifying model paths, hardening evaluation input handling (fillna -> str), adding robust probability indexing, and ensuring the evaluation uses the same raw text format the pipeline expects, the canonical saved model achieves:

| Metric      | Class 0 (Fake) | Class 1 (Real) | Overall |
|-------------|---------------:|---------------:|--------:|
| Precision   | 0.76           | 0.70           | 0.72    |
| Recall      | 0.51           | 0.87           | 0.71    |
| F1-score    | 0.61           | 0.77           | 0.70    |
| Support     | 4488           | 5752           | 10240   |
| Accuracy    |                |                | 0.71    |
| Macro avg   | P 0.73 / R 0.69 / F1 0.69 |        |         |
| Weighted avg| P 0.72 / R 0.71 / F1 0.70 |        |         |

Notes:
- This 71% accuracy is measured when feeding the saved pipeline the same raw text input it was trained on (USE_PREPROCESS=False).
- Applying separate/extra preprocessing at evaluation produced much worse results (≈56% accuracy) — fix applied: evaluation now uses raw input by default to match training.
- The primary weakness remains lower recall for class 0 (fake) relative to class 1; model tends toward predicting class 1 more often when given the raw training-format input.

---

## Change log (high level)
- Fixed multiple evaluation bugs: NaN handling in inputs, SettingWithCopyWarning (use df.copy() + .loc), consistent model path via `ml/config.py`.
- Added helpers: `find_model`, `positive_proba`, `map_liar_label` in `ml/utils.py`.
- Replaced fragile predict_proba[:,1] indexing with robust mapping to model.classes_.
- Added debug script `ml/scripts/compare_models_debug.py` to compare raw vs preprocessed inputs and model artifacts.
- Ensured saved model artifact used for canonical evaluation: `ml/models/logreg_grid_best.joblib`.

---

## Interpretation & next steps
- 71% accuracy is a solid baseline for this pipeline. Remaining improvements should focus on:
  - reducing false negatives on class 0 (increase recall for fake), via threshold tuning, class-weight or resampling;
  - richer TF-IDF features (char n-grams, larger max_features), ensembles, or transformer fine-tuning for higher gains.
- See `md/TODO.md` for prioritized tasks to reach 80–90% target.

---