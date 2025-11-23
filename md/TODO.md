# FactFlow AI — Code Tasks (updated 2025-11-23)

## Status summary (quick)
- Current validated accuracy: ~71% on canonical test set (raw input).
- Major fixes done: unified model paths, safe input handling, avoid SettingWithCopyWarning, robust predict_proba handling, debug comparator script, config flag for preprocessing.

---

## ✅ Completed (already done)
- [x] Refactor the `evaluate.py` script: separation of load / preprocess / predict / metrics + debug prints.
- [x] Add error handling for missing columns and wrong dtypes in evaluation.
- [x] Document and enforce input contract (USE_PREPROCESS flag; pipeline expects raw text by default).
- [x] Fix SettingWithCopyWarning in preprocessing (use df.copy() + .loc).
- [x] Add `ml/config.py` with canonical MODEL_PATH and LEGACY_MODEL_PATHS.
- [x] Add helpers in `ml/utils.py`: find_model, positive_proba, map_liar_label, updated preprocess semantics.
- [x] Create `ml/scripts/compare_models_debug.py` to compare raw vs preprocessed behavior and model artifacts.
- [x] Replace hardcoded model loads and fragile predict_proba[:,1] usages in scripts (save_probs, check_probs, compare_pipeline_input).
- [x] Ensure evaluator tolerates NaNs (fillna/str) before vectorizer.
- [x] Produce reproducible evaluation showing 71% accuracy.

## Removed / Not needed
- [x] "Drop direct joblib.load usages without config" — replaced by config+find_model approach.
- [x] Duplicate/legacy pipeline files left unused — keep them archived; don't use as canonical artifacts.

---

## TODO: Short-term fixes (next 1–3 days) — quick wins (low effort)
- [ ] Threshold tuning: optimize classification threshold on val set for desired precision/recall tradeoff. (quick)
- [ ] Probability calibration (CalibratedClassifierCV) to make thresholds meaningful. (quick)
- [ ] Add `--skip_train` to `ml/eval.py` or use `python -m ml.evaluate` to evaluate saved models without retraining. (done/verify)
- [ ] Save evaluation metrics (JSON) alongside model artifact after each run. (quick)
- [ ] Add a small script to export top misclassified examples and top tokens for manual inspection. (quick)

## TODO: Mid-term (this week) — moderate effort, expected ROI
- [ ] Hyperparameter sweep for TF-IDF + Logistic: extend grid (ngram ranges, char n-grams, higher max_features, min_df). (medium)
- [ ] Stratified k-fold cross-validation to stabilize CV estimates and grid search. (medium)
- [ ] Class imbalance experiments: class_weight='balanced', upsample minority in train, focal loss (if using other models). (medium)
- [ ] Feature expansion: add char n-grams, word n-grams (1-3), TF-IDF max_df/min_df scans. (medium)
- [ ] Ensemble baseline: train a LightGBM or RandomForest on TF-IDF features and compare. (medium)

## TODO: Longer-term (next week) — higher effort, larger gains
- [ ] Fine-tune a transformer (DistilBERT/roberta-base) on your dataset (likely +5–15% absolute). Prepare GPU run / experiment script. (long)
- [ ] Domain-adaptive pretraining or continued pretraining if domain text is specialized. (long)
- [ ] Create CI for training/eval and unit tests for loading/predicting pipelines. (long)
- [ ] Model registry / versioning: save model + metrics + config in a simple registry (folder per model with JSON). (long)
- [ ] Dockerize inference service (FastAPI) with health checks, input validation, and logging. (long)

---

## Targeted plan to reach 80–90% accuracy (prioritized actions)
Priority 1 (do first; highest ROI)
- [ ] Retrain with richer TF-IDF features (add char n-grams up to 3, word n-grams 1-3, increase max_features to 50k) + run stratified CV. (expected +2–5%)
- [ ] Tune class weights / threshold per class to reduce missed class-0 (improve recall for class 0). (expected +1–3% on F1/recall)
- [ ] Calibrate probabilities and choose threshold optimizing business metric (precision@k or F1). (improves decision quality)

Priority 2 (if still needed)
- [ ] Ensemble TF-IDF Logistic + tree-based on TF-IDF features (stack or simple averaging). (expected +1–3%)
- [ ] Clean label noise: sample and relabel ambiguous / noisy examples discovered in misclassified_top50. (quality improvement)

Priority 3 (higher compute / time)
- [ ] Fine-tune DistilBERT or similar on your dataset; use validation to select best checkpoint. (expected major uplift, +5–15%)
- [ ] If successful, consider knowledge distillation to a smaller model for production latency.

Acceptable deployment targets (suggested)
- Low-risk: accuracy >= 75% and balanced class F1 >= 0.7
- Typical: accuracy >= 80% and per-class recall/precision acceptable to product
- High-confidence production: accuracy >= 85% with calibrated probabilities and SLA/monitoring

---

## Engineering & production-readiness tasks
- [ ] Add unit tests for evaluate, predict path, and utils functions (pytests).
- [ ] Add reproducible run script with seed, requirements.txt, and env details.
- [ ] Add monitoring plan: drift detection, daily metric logs, alerting on performance drop.
- [ ] Document input contract in README and add a validation function in API to enforce it.

---

## Next-week sprint (concrete plan)
Day 1:
- Run targeted TF-IDF feature sweep + stratified CV. Save best model artifacts and metrics JSON.

Day 2:
- Threshold tuning + probability calibration. Export calibrated model and chosen threshold; update eval scripts.

Day 3:
- Error analysis: review top-100 misclassified samples, correct label noise / make a small curated dataset for fine-tuning.

Day 4–5:
- If TF-IDF improvements not enough, start Transformer fine-tune experiments (prepare scripts/Docker/GPU).

Deliverables for next week:
- Metric report (CSV/JSON) comparing baseline vs improved models
- Model artifact + metrics saved in ./models/<name> with README note
- Small PR with changes + unit tests for critical utils

---

## Suggested commit message
- "docs: update TODO with completed tasks and 80-90% roadmap; add prioritized training & production tasks"