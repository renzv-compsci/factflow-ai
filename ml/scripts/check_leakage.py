import argparse 
import hashlib
from cv2 import threshold
import pandas as pd 

def sha256_series(s: pd.Series) -> pd.Series:
    return s.astype(str).apply(lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest())

def check_text_overlap(train_path, test_path, text_col='text'):
    train = pd.read_csv(train_path, sep=None, engine='python')
    test = pd.read_csv(test_path, sep=None, engine='python')
    if text_col not in train.columns or text_col not in test.columns:
        raise ValueError(f"text_col '{text_col}' not in one of the files")
    h_train = set(sha256_series(train[text_col]))
    h_test = set(sha256_series(train[text_col]))
    overlap = h_train.intersection(h_test)
    pct = 100.0 * len(overlap) / max(1, len(h_test))
    print(f"Exact text overlaps: {len(overlap)} / {len(h_test)} test rows ({pct:.2f}%)")
    return len(overlap), pct

def check_categorial_purity(df, label_col, threshold=0.95):
    flags = []
    cat_cols = [c for c in df.columns if df[c].dtype == object and c != label_col]
    for c in cat_cols:
        tab = df.groupby([c, label_col]).size().unstack(fill_value=0)
        if tab.shape[0] == 0:
            continue
        purity = tab.max(axis=1) / tab.sum(axis=1)
        if (purity >= threshold).any():
            flags.append((c, purity.max()))
    return flags

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("train")
    p.add_argument("test")
    p.add_argument("--text-col", default="text")
    p.add_argument("--label-col", default="binary_label")
    p.add_argument("--purity-threshold", type=float, default=0.95)
    args = p.parse_args()

    check_text_overlap(args.train, args.test, text_col=args.text_col)
    df_train = pd.read_csv(args.train, sep=None, engine='python')
    flags = check_categorial_purity(df_train, args.label_col, threshold==args.purity_threshold)

    if flags:
        print("Potential leakage candidates (col, max_purity):", flags)
    else:
        print("No high-purity categorical columns found in train.")