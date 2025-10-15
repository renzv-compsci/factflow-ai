# FactFlow AI — Code Tasks for Today

## TODO Date: 2025-10-06

---

## 🛠️ Code-Focused Action Steps

### 1. **Model Evaluation Pipeline**
- [/] Refactor the `evaluate.py` script:
  - Ensure clean separation between data loading, preprocessing, prediction, and metrics reporting.
  - Add debug print statements for dataframe columns and sample data to aid troubleshooting.
  - Handle both index-based and name-based column references for flexibility.
- [ ] Implement automated tests for the evaluation pipeline using sample datasets.
- [/] Add error handling for common issues (e.g., missing columns, wrong data types).

### 2. **Feature Engineering & Data Handling**
- [ ] Review all feature columns for possible data leakage.
- [/] Document and enforce consistent preprocessing steps for both training and test sets.
- [/] Check for and handle class imbalance programmatically (e.g., via upsampling, downsampling, or scikit-learn’s `class_weight`).

### 3. **Model Training & Validation**
- [/] Integrate k-fold cross-validation in the training script to assess generalization.
- [/] Add regularization options (L1/L2) to training.
- [/] Train and compare multiple algorithms (e.g., Logistic Regression, Random Forest, SVM).

## TODOs to Reach ≥85% Accuracy

- [ ] **Check Data Balance**
   - [ ] Examine the ratio of fake to real news in both training and test sets.
   - [ ] Consider oversampling the minority class or undersampling the majority class to balance the data.

- [ ] **Improve Data Quality**
   - [ ] Clean and normalize text data more thoroughly (remove stopwords, punctuation, lemmatize/stem words).
   - [ ] Remove duplicate or near-duplicate articles.

- [ ] **Feature Engineering**
   - [ ] Experiment with additional text features: n-grams, TF-IDF settings, or word embeddings (Word2Vec, FastText, or BERT).
   - [ ] Try adding metadata features if available (source, author, date).

- [ ] **Model Selection/Tuning**
   - [ ] Try other algorithms: Random Forest, SVM, XGBoost, or deep learning models (LSTM, transformers).
   - [ ] Perform hyperparameter tuning (e.g., using GridSearchCV) for better regularization, C, penalty, etc.

- [ ] **Cross-Validation**
   - [ ] Use stratified k-fold cross-validation to ensure robust model evaluation and tuning.
   - [ ] Monitor precision and recall for both classes during validation.

- [ ] **Address Class Imbalance in Training**
   - [ ] Use `class_weight='balanced'` in model training.
   - [ ] Augment data for the underrepresented class.

- [ ] **Evaluate Data Leakage**
   - [ ] Double-check that there is no overlap between train and test sets.
   - [ ] Ensure proper shuffling and splitting.

- [ ] **Advanced Approaches**
   - [ ] Try ensemble methods (e.g., voting classifiers, stacking).
   - [ ] Incorporate pretrained language models (DistilBERT, RoBERTa) for richer text representation.

- [ ] **Human-in-the-loop**
   - [ ] For a production system, consider flagging low-confidence predictions for manual review.

- [ ] **Monitor and Iterate**
   - [ ] Track model performance over time as you apply improvements.
   - [ ] Periodically retrain the model with fresh, diverse data.

---

**Goal:**  
> Achieve **at least 85% accuracy** (and balanced F1-scores for both classes) for reliable fake news detection.

---