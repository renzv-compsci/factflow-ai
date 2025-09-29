import pandas as pd 

filepath = "ml/data/test_set/train.tsv"
df = pd.read_csv(filepath, sep="\t", header=None)

# count label
label_counts = df[1].value_counts()
print("Label counts:")
print(label_counts)

# random samples 
print("\nRandom samples:")
print(df.sample(5))