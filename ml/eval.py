import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from ml.config import USE_PREPROCESS
from ml.utils import clean_text

def main(data_path, text_col, label_col, out_model, random_state=42):
    df = pd.read_csv(data_path, sep="\t", engine="python")

    # allow numeric name for text_col 
    try: 
        tc = int(text_col)
        X = df.iloc[:, tc].astype(str)
    except Exception:
        X = df[text_col].astype(str)
    y = df[label_col].astype(int)

    if USE_PREPROCESS:
        X = X.apply(clean_text)

    X_train, X_val, y_train, y_val = train_test_split (X, y, test_size=0.2, random_state=random_state, stratify=y)

    # model pipeline 
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    # grids 
    param_grid = {
        "tfidf__ngram_range": [(1,1), (1,2)],
        "tfidf__max_features": [10000, 30000],
        "tfidf__min_df": [1,2],
        "clf__C": [0.1, 1.0, 5.0],
        "clf__class_weight": [None, "balanced"]
    }

    # utilize grid search
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=2)
    gs.fit(X_train, y_train)

    # find best params 
    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    # probabilities of best parameters 
    probs = best.predict_proba(X_val)[:, 1]
    preds = best.predict(X_val)
    
    print("Validation ROC AUC:", roc_auc_score(y_val, probs))
    print("Classification Report (val):")
    print(classification_report(y_val, preds, zero_division=0))

    out_model = Path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, out_model)
    print("Saved best model to:", out_model)

    # save grid results for preview 
    results_df = pd.DataFrame(gs.cv_results_)
    results_df.to_csv(out_model.with_suffix(".grid_results.csv"), index=False)
    print("Saved grid results to:", out_model.with_suffix(".grid_results.csv"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="ml/data/test_set/train_mapped_clean.tsv")
    p.add_argument("--text_col", default="2", help="text column name or index")
    p.add_argument("--label_col", default="binary_label")
    p.add_argument("--out_model", default="ml/models/logreg_grid_best.joblib")
    args = p.parse_args()
    main(args.data, args.text_col, args.label_col, args.out_model)
    