import pandas as pd
df = pd.read_csv("ml/data/test_set/train_mapped.tsv", sep="\t", engine="python")
y = df["binary_label"].astype(int)
print("counts:\n", y.value_counts())
print("majority acc:", max(y.value_counts(normalize=True)))
