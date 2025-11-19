# save_probs.py
from joblib import load
import pandas as pd, numpy as np
m = load('logreg_pipeline.joblib')
df = pd.read_csv('ml/data/test_set/train_mapped.tsv', sep='\t', engine='python')
X = df.iloc[:, int('2')]
if hasattr(m, 'predict_proba'):
    probs = m.predict_proba(X)[:,1]
else:
    scores = m.decision_function(X)
    probs = 1/(1+np.exp(-scores))
df['p_pos'] = probs
df.to_csv('test_with_probs.csv', index=False)
print("Saved test_with_probs.csv")