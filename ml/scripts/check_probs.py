from joblib import load 
import pandas as pd 
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np 

m = load('logreg_pipeline.joblib')
df = pd.read_csv('ml/data/test_set/train_mapped.tsv', sep='\t', engine='python')
X = df.iloc[:, int('2')]
y = df['binary_label'].astype(int)

# get probs, handle both cases 
if hasattr(m, 'predict_proba'):
    probs = m.predict_proba(X)[:, 1]
else:
    scores = m.decision_function(X)
    probs = 1 /(1 + np.exp(-scores))

print("mean p(class1) | true=1:" , probs[y==1].mean())
print("mean p(class1) | true=0:", probs[y==1].mean())
print("ROC AUC:", roc_auc_score(y, probs))

# examine metrics at diff thresholds 
for thr in (0.5, 0.3, 0.2, 0.1):
    y_pred = (probs >= thr).astype(int)
    print(f"\nThreshold {thr}")
    print(classification_report(y, y_pred))