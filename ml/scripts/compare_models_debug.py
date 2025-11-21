import sys
from pathlib import Path
import joblib
import pandas as pd 
from sklearn.metrics import classification_report
from ml.config import LEGACY_MODEL_PATHS
from ml.utils import find_model, positive_proba, map_liar_label, clean_text

# load model 
def load_model_or_none(path):
    try:
        return joblib.load(path)
    except Exception as e: 
        print(f"Cannot load {path}: {e}")
        return None 

# get score and report 
def score_and_report(model, X_raw, X_pre, y_true, label="model"):
    print(f"\n--- {label} ---")
    print("classes_:", getattr(model, "classes_", None))

    # model.predict if available
    try: 
        preds = model.predict(X_raw)
        print("predict() counts:", pd.Series(preds).value_counts().to_dict())
        print("classification_report (predict on raw):")
        print(classification_report(y_true, preds, zero_division=0))
    except Exception as e:
        print("predict() error:", e)

    # safe proba on raw 
    try: 
        p_raw = positive_proba(model, X_raw, positive_label=1)
        preds_thr = (p_raw >= 0.5).astype(int)
        print("classification_report (proba >= 0.5 on raw):")
        print(classification_report(y_true, preds_thr, zero_division=0))
    except Exception as e:
        print("proba(raw) error:", e)

    # proba on preprocessed text
    try: 
        p_pre = positive_proba(model, X_pre, positive_label=1)
        preds_pre = (p_pre >= 0.5).astype(int)
        print("classification_report (proba >= 0.5 on preprocessed text):")
        print(classification_report(y_true, preds_pre, zero_division=0))
    except Exception as e: 
        print("proba(pre) error:", e)

def main():
    data_path = Path("ml/data/test_set/train_mapped_clean.tsv")
    if not data_path.exists():
        data_path = Path("ml/data/test_set/train_mapped.tsv")
    df = pd.read_csv(data_path, sep="\t", engine="python")
    print("data path used:", data_path)
    print("rows,cols:", df.shape)
    # choose text column as before (index 2)
    text_col_idx = 2
    X_raw = df.iloc[:, text_col_idx].astype(str)
    # create preprocessed copy (same logic your pipeline uses)
    X_pre = X_raw.apply(clean_text)
    y = df["binary_label"].astype(int)

    # find a couple model paths to compare
    candidates = LEGACY_MODEL_PATHS.copy()
    # also include the grid-best path if different
    candidates.append("ml/models/logreg_grid_best.joblib")

    for cand in candidates:
        if Path(cand).exists():
            m = load_model_or_none(cand)
            if m is None:
                continue
            score_and_report(m, X_raw, X_pre, y, label=cand)

if __name__ == "__main__":
    main()