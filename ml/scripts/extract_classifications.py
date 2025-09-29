import pandas as pd 

filepath = "ml/data/test_set/train.tsv"

df = pd.read_csv(filepath, sep="\t", header=None)

labels = df[1]
unique_labels = labels.unique()

print("Unique classifications found in train.tsv:")
for label in unique_labels:
    print(label)