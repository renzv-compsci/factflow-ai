# FactFlow AI — Code Tasks for Today

## TODO Date: 2025-10-06

---

## 🛠️ Code-Focused Action Steps

### 1. **Model Evaluation Pipeline**
- [/] Refactor the `evaluate.py` script:
  - Ensure clean separation between data loading, preprocessing, prediction, and metrics reporting.
  - Add debug print statements for dataframe columns and sample data to aid troubleshooting.
  - Handle both index-based and name-based column references for flexibility.
- [/] Implement automated tests for the evaluation pipeline using sample datasets.
- [ ] Add error handling for common issues (e.g., missing columns, wrong data types).

### 2. **Feature Engineering & Data Handling**
- [/] Review all feature columns for possible data leakage.
- [/] Document and enforce consistent preprocessing steps for both training and test sets.
- [ ] Check for and handle class imbalance programmatically (e.g., via upsampling, downsampling, or scikit-learn’s `class_weight`).

### 3. **Model Training & Validation**
- [/] Integrate k-fold cross-validation in the training script to assess generalization.
- [ ] Add regularization options (L1/L2) to training.
- [ ] Train and compare multiple algorithms (e.g., Logistic Regression, Random Forest, SVM).