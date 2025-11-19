import sys
import traceback
from joblib import load 
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report

def find_model(paths):
    for p in paths: 
        try: 
            return load(p), p
        except Exception:
            continue
    return None, None 

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))

def get_probs(model, X):
    """
    Try predict_proba, then decision_function. Return (1D numpy probs, None) on success,
    or (None, error_message) on failure. Validate returned length == nrows(X).
    """
    n_rows = getattr(X, "shape", (len(X),))[0]
    try:
        probs_raw = model.predict_proba(X)
        probs_arr = np.asarray(probs_raw)
        # Normalize to 1D probability for positive class
        if probs_arr.ndim == 2 and probs_arr.shape[1] >= 2:
            probs = probs_arr[:, 1]
        elif probs_arr.ndim == 2 and probs_arr.shape[1] == 1:
            probs = probs_arr.ravel()
        elif probs_arr.ndim == 1:
            probs = probs_arr
        else:
            return None, f"predict_proba returned unexpected shape: {probs_arr.shape}"
        if probs.shape[0] != n_rows:
            return None, f"predict_proba produced {probs.shape[0]} scores but input has {n_rows} rows"
        return probs, None
    except Exception:
        tb1 = traceback.format_exc()
        try:
            scores = model.decision_function(X)
            scores_arr = np.asarray(scores)
            # flatten if necessary
            if scores_arr.ndim > 1:
                # prefer treating multi-col scores as per-row by taking last column if shape matches
                if scores_arr.shape[0] == n_rows:
                    scores_flat = scores_arr[:, -1]
                else:
                    scores_flat = scores_arr.ravel()
            else:
                scores_flat = scores_arr
            probs = _sigmoid(scores_flat)
            if probs.shape[0] != n_rows:
                return None, f"decision_function produced {probs.shape[0]} scores but input has {n_rows} rows"
            return probs, None
        except Exception:
            tb2 = traceback.format_exc()
            return None, f"predict_proba error:\n{tb1}\n---\ndecision_function error:\n{tb2}"

def summarize(probs, y, label):
    if probs is None:
        return f"{label}: model cannot score this input shape\n"
    # ensure y and probs have same length
    if len(probs) != len(y):
        return f"{label}: length mismatch: probs={len(probs)} vs labels={len(y)}"
    out = []
    out.append(f"{label}: mean p_pos overall: {np.mean(probs):.6f}")
    # guard empty slices
    try:
        out.append(f"{label}: mean p_pos | true=1: {np.mean(probs[np.asarray(y)==1]):.6f}")
    except Exception as e:
        out.append(f"{label}: mean p_pos | true=1: error {e}")
    try:
        out.append(f"{label}: mean p_pos | true=0: {np.mean(probs[np.asarray(y)==0]):.6f}")
    except Exception as e:
        out.append(f"{label}: mean p_pos | true=0: error {e}")

    try: 
        out.append(f"{label}: ROC AUC: {roc_auc_score(y,probs):.6f}")
    except Exception as e:
        out.append(f"{label}: ROC AUC: error {e}")
    for thr in (0.5, 0.3, 0.2, 0.1):
        y_pred = (probs >= thr).astype(int) 
        out.append(f"\n{label} - threshold {thr}\n" + classification_report(y, y_pred, zero_division=0))
    return "\n".join(out)

def main(model_paths=None, data_path='ml/data/test_set/train_mapped.tsv', text_col='2', label_col='binary_label'):
    if model_paths is None: 
        model_paths = ['ml/models/logreg_pipeline.joblib', 'logreg_pipeline.joblib', 'models/logreg_pipeline.joblib']
    model, used = find_model(model_paths)
    if model is None: 
        print("No model found in paths:", model_paths)
        sys.exit(1)
    print("Loaded model from:", used)

    df = pd.read_csv(data_path, sep='\t', engine='python')
    # allow numeric index or name 
    try: 
        text_idx = int(text_col)
        text_is_index = True 
    except Exception:
        text_is_index = False 
        text_idx = text_col

    variants = []
    if text_is_index:
        variants.append(("Series_iloc", df.iloc[:, text_idx]))
        variants.append(("DataFrame_iloc", df.iloc[:, [text_idx]]))
        try: 
            col_label = df.columns[text_idx]
            variants.append(("DataFrame_collabel", df[[col_label]]))
        except Exception:
            pass
    else: 
        variants.append(("Series_label", df[text_idx]))
        variants.append(("DataFrame_label", df[[text_idx]]))
    y = df[label_col].astype(int)

    out_lines = [f"Model: {used}", f"Data columns: {list(df.columns)[:20]} (showing first 20)"]
    for name, X in variants:
        out_lines.append("\n--- Variant: " + name + " ---") 

        try: 
            out_lines.append(f"Input shape: {getattr(X, 'shape', 'unknown')}, type: {type(X)}")
            if hasattr(X, 'dtypes'):
                out_lines.append(f"Column dtypes: {X.dtypes.to_dict()}")
        except Exception:
            pass

        probs, err = get_probs(model, X)
        if err: 
            out_lines.append("ERROR when scoring input:\n" + err)
            out_lines.append("\n")
        if probs is None:
            out_lines.append(f"{name}: model cannot score this input shape")
        else:
            out_lines.append(summarize(probs, y, name))

    report = "\n".join(out_lines)
    print(report)
    with open('ml/scripts/compare_input_results.txt', 'w', encoding='utf-8') as f:
        f.write(report)
   
if __name__ == '__main__':
    # defaults mirror your earlier runs; override with args: [model_path] [data_path] [text_col] [label_col]
    args = sys.argv[1:]
    if len(args) >= 1:
        model_paths = [args[0]]
    else:
        model_paths = None
    data_path = args[1] if len(args) >= 2 else 'ml/data/test_set/train_mapped.tsv'
    text_col = args[2] if len(args) >= 3 else '2'
    label_col = args[3] if len(args) >= 4 else 'binary_label'
    main(model_paths=model_paths, data_path=data_path, text_col=text_col, label_col=label_col)