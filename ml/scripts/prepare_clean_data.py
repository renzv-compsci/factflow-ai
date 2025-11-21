import pandas as pd 

df = pd.read_csv("ml/data/test_set/train_mapped.tsv", sep="\t", engine="python")
print("cols before:", df.columns.tolist())

df_clean = df.drop(columns=[c for c in df.columns if c in['0', '1', 'id', 'filename']], errors='ignore')
print("cols after:", df_clean.columns.tolist())
out = "ml/data/test_set/train_mapped_clean.tsv"

df_clean.to_csv(out, sep="\t", index=False)
print("Wrote cleaned file:", out)