# save_probs.py
from joblib import load
import pandas as pd, numpy as np
from ml.config import LEGACY_MODEL_PATHS
from ml.utils import find_model, positive_proba

mp = find_model(LEGACY_MODEL_PATHS)
if mp is None:
    raise FileNotFoundError(f"No model found in {LEGACY_MODEL_PATHS}")
m = load(mp)
print("Using model:", mp, "classes:", getattr(m, "classes_", None))


df = pd.read_csv('ml/data/test_set/train_mapped.tsv', sep='\t', engine='python')
X = df.iloc[:, int('2')]
if hasattr(m, 'predict_proba'):
    probs = positive_proba(m, X, positive_label=1)
else:
    scores = m.decision_function(X)
    probs = 1/(1+np.exp(-scores))
df['p_pos'] = probs
df.to_csv('test_with_probs.csv', index=False)
print("Saved test_with_probs.csv")