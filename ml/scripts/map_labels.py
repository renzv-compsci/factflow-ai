import pandas as pd 

filepath = "ml/data/test_set/train.tsv"
output_path = "ml/data/test_set/train_mapped.tsv"

df = pd.read_csv(filepath, sep="\t", header=None)

def map_liar_label(label):
    label = str(label).lower().strip()
    if label in ['false', 'pants-fire', 'barely-true']:
        return 0 # fake 
    elif label in ['true', 'mostly-true', 'half-true']:
        return 1 # real 
    else:
        return None 
    
# mapping to label column (index 1)
df['binary_label'] = df[1].apply(map_liar_label)

# save to new tsv
df.to_csv(output_path, sep="\t", index=False)
print("Saved mapped labels to", output_path)

