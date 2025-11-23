# save_probs.py
from joblib import load
import pandas as pd, numpy as np
from ml.config import USE_PREPROCESS, LEGACY_MODEL_PATHS
from ml.utils import find_model, clean_text, positive_proba

mp = find_model(LEGACY_MODEL_PATHS)
if mp is None:
    raise FileNotFoundError(f"No model found in {LEGACY_MODEL_PATHS}")
m = load(mp)
print("Using model:", mp, "classes:", getattr(m, "classes_", None))

df = pd.read_csv('ml/data/test_set/train_mapped.tsv', sep='\t', engine='python')

text_col = '2'
X = df.iloc[:, int(text_col)].astype(str)

if USE_PREPROCESS:
    X_for_model = X.apply(clean_text)
else: 
    X_for_model = X

# get prob 
if hasattr(m, 'predict_proba'):
    probs = positive_proba(m, X_for_model, positive_label=1)
elif hasattr(m, 'decision_function'):
    scores = m.decision_function(X_for_model)
    probs = 1.0 / (1.0 + np.exp(-scores))
else:
    # fallback
    preds = m.predict(X_for_model)
    probs = np.array(preds, dtype=float)
    
df['p_pos'] = probs
df.to_csv('test_with_probs.csv', index=False)
print("Saved test_with_probs.csv")