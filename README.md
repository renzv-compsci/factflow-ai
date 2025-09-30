# factflow-ai
A simple AI-powered Fake News Detector web app. Users can paste articles or links to check the authenticity of news using a machine learning model. The app provides instant results with confidence scores, saves query history, and features a modern, tech-inspired UI.


# Model Evaluation Report

## Overview

This document summarizes the performance of the trained fake news classifier as evaluated on a held-out test set. It includes a detailed interpretation of the classification report, an analysis of model weaknesses such as bias and overfitting, and discusses possible data leakage and distributional differences between training and testing.

---

## Classification Report Summary

The following metrics were obtained from evaluating the model on the test set:

```
              precision    recall  f1-score   support

           0       0.44      0.95      0.60      4488
           1       0.63      0.07      0.12      5752

    accuracy                           0.45     10240
   macro avg       0.53      0.51      0.36     10240
weighted avg       0.55      0.45      0.33     10240
```

- **Accuracy:** 45% (Only 45% of test samples are correctly classified)

### Class 0 (Fake News)
- **Precision (0.44):** When the model predicts "fake", it is correct 44% of the time.
- **Recall (0.95):** Of all actual fake news samples, the model correctly identifies 95%.
- **F1-score (0.60):** The harmonic mean of precision and recall shows moderate performance.

### Class 1 (True News)
- **Precision (0.63):** When the model predicts "true", it is correct 63% of the time.
- **Recall (0.07):** Of all actual true news samples, the model only identifies 7%.
- **F1-score (0.12):** Very low, indicating poor performance on this class.

---

## Interpretation

### 1. **Model Bias**
- The model is heavily biased toward predicting "fake news" (class 0).
- It successfully identifies almost all fake news but fails to recognize true news, misclassifying most of them as fake.

### 2. **Overfitting**
- Training accuracy is very high (99%), while test accuracy is much lower (45%).
- This suggests the model memorized patterns in the training data instead of learning generalizable features.
- Overfitting occurs when a model is too complex or the training process doesn't include enough regularization.

### 3. **Data Leakage**
- Data leakage happens when information from outside the training dataset is used to create the model, leading to artificially high performance.
- Possible causes in this context:
  - Inclusion of columns/features in training that directly or indirectly reveal the label.
  - Preprocessing steps that are inconsistent between training and test sets.
- If leakage occurred, the model may learn shortcuts that don't exist in real-world unseen data.

### 4. **Distributional Differences**
- If the training and test sets differ significantly in their distributions (e.g., different proportions of fake and true news, or different topics/authors), the model may not generalize.
- Class imbalance can contribute to the model's tendency to predict the majority class.

---

## Recommendations

1. **Review Features for Leakage**
   - Ensure no columns directly encode the label (e.g., avoid using `binary_label` or derived features).
   - Align preprocessing for both train and test sets.

2. **Address Model Bias**
   - Investigate class imbalance and consider rebalancing techniques (e.g., upsampling, downsampling, class weights).

3. **Reduce Overfitting**
   - Apply regularization (L1/L2) to your model.
   - Use cross-validation to assess generalizability.

4. **Monitor Data Splitting**
   - Ensure random, stratified splitting of train/test sets, with no overlap.

5. **Model Improvements**
   - Experiment with different algorithms.
   - Use more robust feature engineering.
   - Tune hyperparameters.

---

## Conclusion

The current model achieves high accuracy on its training data but poor performance on the test set, predicting "fake news" for most inputs and failing to identify true news reliably. This points to serious overfitting, possible data leakage, and bias issues. Addressing these concerns will be critical for improving the model’s real-world effectiveness.
